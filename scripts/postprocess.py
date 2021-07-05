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

index = faiss.IndexFlatL2(reference.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
index_gpu.add(reference)
reference_dist, reference_ind = index_gpu.search(query, k=10)

submission = pd.DataFrame(columns=['query_id', 'reference_id', 'score'])
submission['query_id'] = np.repeat(query_ids, 10)
submission['reference_id'] = np.array(reference_ids)[reference_ind.ravel()]
submission['score'] = - reference_dist.ravel()

train = np.load('v23/extract/train_feats.npy')

index = faiss.IndexFlatL2(train.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
index_gpu.add(train)
train_dist, train_ind = index_gpu.search(query, k=100)

# matching track
beta = 1.0
tn = 3
s = - train_dist[:, :tn].mean(axis=1) * beta

# submission = pd.read_csv(submission_path.replace('.h5', '.csv'))
query_ind = submission['query_id'].map(lambda x: x[1:]).astype(int).values
submission['score'] -= s[query_ind]
submission.to_csv('v23/extract/tmp.csv', index=False)


# reverse query expansion
index = faiss.IndexFlatIP(train.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
index_gpu.add(train)
query_train_sim, query_train_ind = index_gpu.search(query, k=30)
reference_train_sim, reference_train_ind = index_gpu.search(reference, k=30)

beta = 0.35
k = 10
_query = query - (train[query_train_ind[:, :k]].mean(axis=1) * beta)
# _query = query - (train[query_train_ind[:, :k]] * (query_train_sim[:, :k, None] ** alpha)).mean(axis=1) * beta
_query /= np.linalg.norm(query, axis=1, keepdims=True)
_reference = reference - train[reference_train_ind[:, :k]].mean(axis=1) * beta
_reference /= np.linalg.norm(reference, axis=1, keepdims=True)

index = faiss.IndexFlatL2(_reference.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
index_gpu.add(_reference)
reference_dist, reference_ind = index_gpu.search(_query, k=10)

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
