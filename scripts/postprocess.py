import faiss
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def load_descriptor_h5(descs_submission_path):
    """Load datasets from descriptors submission hdf5 file."""

    with h5py.File(descs_submission_path, "r") as f:
        query = f["query"][:]
        reference = f["reference"][:]
        # Coerce IDs to native Python unicode string no matter what type they were before
        query_ids = np.array(f["query_ids"][:], dtype=object).astype(str).tolist()
        reference_ids = np.array(f["reference_ids"][:], dtype=object).astype(str).tolist()

        if "train" in f:
            train = f["train"][:]
        else:
            train = None

    return query, reference, train, query_ids, reference_ids


versions = [
    'v106',
    'v107',
    'v108',
    'v110',
]
qs = []
rs = []
ts = []
for ver in versions:
    _query, _reference, _, query_ids, reference_ids = load_descriptor_h5(f'{ver}/extract/fb-isc-submission.h5')
    _train = np.load(f'{ver}/extract/train_feats.npy')
    qs.append(_query)
    rs.append(_reference)
    ts.append(_train)

query = np.concatenate(qs, axis=1)
reference = np.concatenate(rs, axis=1)
train = np.concatenate(ts, axis=1)

# query /= np.linalg.norm(query, axis=1, keepdims=True)
# reference /= np.linalg.norm(reference, axis=1, keepdims=True)
# train /= np.linalg.norm(train, axis=1, keepdims=True)

# pca = TruncatedSVD(n_components=256, random_state=0)
# pca.fit(train)
# query = pca.transform(query).astype('float32')
# reference = pca.transform(reference).astype('float32')
# train = pca.transform(train).astype('float32')

# query /= np.linalg.norm(query, axis=1, keepdims=True)
# reference /= np.linalg.norm(reference, axis=1, keepdims=True)
# train /= np.linalg.norm(train, axis=1, keepdims=True)

# out = f'../output/cat_norm_pca_norm_eval.h5'
# with h5py.File(out, 'w') as f:
#     f.create_dataset('query', data=query)
#     f.create_dataset('reference', data=reference)
#     f.create_dataset('train', data=train)
#     f.create_dataset('query_ids', data=np.array(query_ids, dtype='S6'))
#     f.create_dataset('reference_ids', data=np.array(reference_ids, dtype='S7'))

# query, reference, train, query_ids, reference_ids = load_descriptor_h5(f'../output/cat_norm_pca_norm_eval.h5')

index_train = faiss.IndexFlatIP(train.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
index_train.add(train)

sim, ind = index_train.search(train, k=10)
k = 10
alpha = 3.0
_train = (train[ind[:, :k]] * (sim[:, :k, None] ** alpha)).sum(axis=1)
_train /= np.linalg.norm(_train, axis=1, keepdims=True)

index_train = faiss.IndexFlatIP(train.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
index_train.add(_train)

def embedding_isolation(embedding, train, index, beta, k, num_iter):
    for _ in range(num_iter):
        _, ind = index.search(embedding, k=k)
        embedding = embedding - (train[ind[:, :k]].mean(axis=1) * beta)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding.astype('float32')


q_beta = 0.35
q_k = 10
q_num_iter = 1

r_beta = 0.35
r_k = 10
r_num_iter = 1

_query = query
_reference = reference
_query = embedding_isolation(_query, _train, index_train, q_beta, q_k, q_num_iter)
_reference = embedding_isolation(_reference, _train, index_train, r_beta, r_k, r_num_iter)

index_reference = faiss.IndexFlatL2(_reference.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_reference = faiss.index_cpu_to_all_gpus(index_reference, co=co, ngpu=ngpu)
index_reference.add(_reference)
reference_dist, reference_ind = index_reference.search(_query, k=10)

out = f'../exp/{versions[0]}/extract/{versions[0]}_iso.h5'
with h5py.File(out, 'w') as f:
    f.create_dataset('query', data=_query)
    f.create_dataset('reference', data=_reference)
    f.create_dataset('query_ids', data=np.array(query_ids, dtype='S6'))
    f.create_dataset('reference_ids', data=np.array(reference_ids, dtype='S7'))

submission = pd.DataFrame(columns=['query_id', 'reference_id', 'score'])
submission['query_id'] = np.repeat(query_ids, 10)
submission['reference_id'] = np.array(reference_ids)[reference_ind.ravel()]
submission['score'] = - reference_dist.ravel()
submission.to_csv(out.replace('.h5', '.csv'), index=False)


beta = 0.5
tn = 3
query_train_sim, query_train_ind = index_train.search(_query, tn)
# reference_train_sim, reference_train_ind = index_train.search(_reference, tn)
sq = query_train_sim[:, :tn].mean(axis=1)
# sr = reference_train_sim[:, :tn].mean(axis=1)

submission = pd.DataFrame(columns=['query_id', 'reference_id', 'score'])
submission['query_id'] = np.repeat(query_ids, 10)
submission['reference_id'] = np.array(reference_ids)[reference_ind.ravel()]
submission['score'] = - reference_dist.ravel()

query_ind = submission['query_id'].map(lambda x: x[1:]).astype(int).values
# reference_ind = submission['reference_id'].map(lambda x: x[1:]).astype(int).values
submission['score'] -= sq[query_ind] * beta
submission.to_csv(f'../exp/{versions[0]}/extract/{versions[0]}_iso_norm.csv', index=False)
