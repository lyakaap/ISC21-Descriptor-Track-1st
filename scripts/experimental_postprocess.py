# similarity活用版
def embedding_isolation(embedding, train, index, beta, k, num_iter, threshold):
    for _ in range(num_iter):
        sim, ind = index.search(embedding, k=k)
        weights = sim[:, :k, None] > threshold
        negative_embeddings = (train[ind[:, :k]] * weights).sum(axis=1) / (weights.sum(axis=1) + 1e-9)
        # embedding = embedding - (train[ind[:, :k]] * (sim[:, :k, None] ** alpha)).mean(axis=1) * beta
        embedding = embedding - negative_embeddings * beta
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding.astype('float32')

# clustering db
num_clusters = 10000
clus = faiss.Clustering(train.shape[1], num_clusters)
clus.seed = np.random.randint(1234)
clus.niter = 20
clus.max_points_per_centroid = 10000
clus.train(train, index_train)
centroids = faiss.vector_float_to_array(clus.centroids)
centroids = centroids.reshape(num_clusters, train.shape[1])
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

index_centroids = faiss.IndexFlatIP(centroids.shape[1])
ngpu = faiss.get_num_gpus()
co = faiss.GpuMultipleClonerOptions()
co.shard = False
index_centroids = faiss.index_cpu_to_all_gpus(index_centroids, co=co, ngpu=ngpu)
index_centroids.add(centroids)

num_iter = 1
beta = 0.35
k = 3

_query = query
_reference = reference
_query = embedding_isolation(_query, centroids, index_centroids, beta, k, 1)
_reference = embedding_isolation(_reference, centroids, index_centroids, beta, k, 1)

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
