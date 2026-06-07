import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

SUPPORTED_KNN_METRICS = {"euclidean", "cosine"}

def build_knn_graph(
    X,
    k: int = 10,
    metric: str = "euclidean",
    include_self: bool = False,
    make_undirected: bool = True,
    batch_size: int = 100_000,
):
    """
    Build a kNN graph from feature matrix X.

    Returns:
        edge_index: np.ndarray with shape [2, num_edges]

    edge_index[0] = source nodes
    edge_index[1] = target nodes
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"Expected X with shape [N, F], got {tuple(X.shape)}")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values. Graph construction requires finite features.")

    if k <= 0:
        raise ValueError("k must be positive.")
    
    if metric not in SUPPORTED_KNN_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}. "
            f"Supported metrics are: {sorted(SUPPORTED_KNN_METRICS)}"
        )

    n_nodes = X.shape[0]

    if k >= n_nodes:
        raise ValueError(f"k={k} must be smaller than number of nodes={n_nodes}.")

    n_neighbors = k if include_self else min(k + 1, n_nodes)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto",
        n_jobs=-1,
    )
    nbrs.fit(X)

    src_parts = []
    dst_parts = []

    print("Querying nearest neighbors in batches...")

    for start in tqdm(range(0, n_nodes, batch_size)):
        print(f"Processing nodes {start:,} to {min(start + batch_size, n_nodes):,} / {n_nodes:,}", flush=True)
        end = min(start + batch_size, n_nodes)

        _, indices = nbrs.kneighbors(X[start:end])

        batch_n = end - start
        src_matrix = np.repeat(
            np.arange(start, end, dtype=np.int64)[:, None],
            n_neighbors,
            axis=1,
        )
        dst_matrix = indices.astype(np.int64, copy=False)

        if include_self:
            src = src_matrix.reshape(-1)
            dst = dst_matrix.reshape(-1)
        else:
            non_self = dst_matrix != src_matrix
            rank = np.cumsum(non_self, axis=1)
            keep = non_self & (rank <= k)

            src = src_matrix[keep]
            dst = dst_matrix[keep]

        src_parts.append(src)
        dst_parts.append(dst)

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)

    edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)

    if make_undirected:
        print("Making graph undirected...")

        src = edge_index[0]
        dst = edge_index[1]

        rev_src = dst
        rev_dst = src

        all_src = np.concatenate([src, rev_src])
        all_dst = np.concatenate([dst, rev_dst])

        print("Removing duplicate edges...")

        edges = np.stack([all_src, all_dst], axis=1)
        edges = np.ascontiguousarray(edges)
        edges = np.unique(edges, axis=0)

        edge_index = edges.T.astype(np.int64, copy=False)

    return edge_index