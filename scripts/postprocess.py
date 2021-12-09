import faiss
import h5py
import numpy as np
import pandas as pd


def negative_embedding_subtraction(
    embedding: np.ndarray,
    negative_embeddings: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    num_iter: int = 3,
    k: int = 10,
    beta: float = 0.35,
) -> np.ndarray:
    """
    Post-process function to obtain more discriminative image descriptor.

    Parameters
    ----------
    embedding : np.ndarray of shape (n, d)
        Embedding to be subtracted.
    negative_embeddings : np.ndarray of shape (m, d)
        Negative embeddings to be subtracted.
    faiss_index : faiss.IndexFlatIP
        Index to be used for nearest neighbor search.
    num_iter : int, optional
        Number of iterations. The default is 3.
    k : int, optional
        Number of nearest neighbors to be used for each iteration. The default is 10.
    beta : float, optional
        Parameter for the weighting of the negative embeddings. The default is 0.35.

    Returns
    -------
    np.ndarray of shape (n, d)
        Subtracted embedding.
    """
    for _ in range(num_iter):
        _, topk_indexes = faiss_index.search(embedding, k=k)
        topk_negative_embeddings = negative_embeddings[topk_indexes]

        embedding -= (topk_negative_embeddings.mean(axis=1) * beta)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

    return embedding.astype('float32')


def negative_embedding_subtraction(
    embedding: np.ndarray,
    negative_embeddings: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    num_iter: int = 3,
    k: int = 10,
    beta: float = 0.35,
) -> np.ndarray:
    for _ in range(num_iter):
        _, topk_indexes = faiss_index.search(embedding, k=k)  # search for hard negatives
        topk_negative_embeddings = negative_embeddings[topk_indexes]

        embedding -= (topk_negative_embeddings.mean(axis=1) * beta)  # subtract by hard negative embeddings
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)  # L2-normalize

    return embedding

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


def main():

    versions = [
        'v107',
    ]
    qs = []
    rs = []
    ts = []
    for ver in versions:
        # _query, _reference, _, query_ids, reference_ids = load_descriptor_h5(f'{ver}/extract/{ver}_iso.h5')
        _query, _reference, _, query_ids, reference_ids = load_descriptor_h5(f'{ver}/extract/fb-isc-submission.h5')
        _train = np.load(f'{ver}/extract/train_feats.npy')
        qs.append(_query)
        rs.append(_reference)
        ts.append(_train)

    query = np.concatenate(qs, axis=1)
    reference = np.concatenate(rs, axis=1)
    train = np.concatenate(ts, axis=1)

    index_train = faiss.IndexFlatIP(train.shape[1])
    ngpu = faiss.get_num_gpus()
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False
    index_train = faiss.index_cpu_to_all_gpus(index_train, co=co, ngpu=ngpu)
    index_train.add(train)

    # DBA on training set
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

    q_beta = 0.35
    q_k = 10
    q_num_iter = 1

    r_beta = 0.35
    r_k = 10
    r_num_iter = 1

    _query = query
    _reference = reference
    _query = negative_embedding_subtraction(_query, _train, index_train, q_num_iter, q_k, q_beta)
    _reference = negative_embedding_subtraction(_reference, _train, index_train, r_num_iter, r_k, r_beta)

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


if __name__ == '__main__':
    main()
