import faiss
import h5py
import numpy as np
import pandas as pd


def load_descriptor_h5(descs_submission_path):
    """Load datasets from descriptors submission hdf5 file."""

    with h5py.File(descs_submission_path, "r") as f:
        query = f["query"][:]
        reference = f["reference"][:]
        # Coerce IDs to native Python unicode string no matter what type they were before
        query_ids = np.array(f["query_ids"][:], dtype=object).astype(str).tolist()
        reference_ids = np.array(f["reference_ids"][:], dtype=object).astype(str).tolist()
    return query, reference, query_ids, reference_ids


submission_path = 'v23/extract/fb-isc-submission.h5'
query, reference, query_ids, reference_ids = load_descriptor_h5(submission_path)
train = np.load('v23/extract/train_feats.npy')


index_train = faiss.IndexFlatIP(train.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
index_train.add(train)

def embedding_isolation(embedding, train, index, beta, k, num_iter):
    for _ in range(num_iter):
        _, ind = index.search(embedding, k=k)
        embedding = embedding - (train[ind[:, :k]].mean(axis=1) * beta)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding.astype('float32')

num_iter = 3
beta = 0.35
k = 10

_query = query
_reference = reference
_query = embedding_isolation(_query, train, index_train, beta, k, num_iter)
_reference = embedding_isolation(_reference, train, index_train, beta, k, num_iter)

index_reference = faiss.IndexFlatL2(_reference.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_reference = faiss.index_cpu_to_all_gpus(index_reference, co=co, ngpu=ngpu)
index_reference.add(_reference)
reference_dist, reference_ind = index_reference.search(_query, k=10)

submission = pd.DataFrame(columns=['query_id', 'reference_id', 'score'])
submission['query_id'] = np.repeat(query_ids, 10)
submission['reference_id'] = np.array(reference_ids)[reference_ind.ravel()]
submission['score'] = - reference_dist.ravel()
submission.to_csv('v23/extract/tmp.csv', index=False)

out = f'v23/extract/tmp.h5'
with h5py.File(out, 'w') as f:
    f.create_dataset('query', data=_query)
    f.create_dataset('reference', data=_reference)
    f.create_dataset('query_ids', data=np.array(query_ids, dtype='S6'))
    f.create_dataset('reference_ids', data=np.array(reference_ids, dtype='S7'))


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
submission['score'] -= sq[query_ind]
submission.to_csv('v23/extract/tmp.csv', index=False)
