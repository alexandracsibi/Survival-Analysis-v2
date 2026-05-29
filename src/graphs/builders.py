import numpy as np
from sklearn.neighbors import NearestNeighbors

SUPPORTED_KNN_METRICS = {"euclidean", "cosine"}

def build_knn_graph(
    X,
    k: int = 10,
    metric: str = "euclidean",
    include_self: bool = False,
    make_undirected: bool = True,
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

    _, indices = nbrs.kneighbors(X)

    src_matrix = np.repeat(np.arange(n_nodes, dtype=np.int64)[:, None], n_neighbors, axis=1)
    dst_matrix = indices.astype(np.int64, copy=False)

    if include_self:
        src = src_matrix.reshape(-1)
        dst = dst_matrix.reshape(-1)
    else:
        # Robust selection: remove self-neighbors row-wise and keep the first k remaining neighbors.
        # This avoids the previous reshape bug when some rows had k neighbors and others had k+1.
        non_self = dst_matrix != src_matrix
        rank = np.cumsum(non_self, axis=1)
        keep = non_self & (rank <= k)

        src = src_matrix[keep]
        dst = dst_matrix[keep]

    edge_index = np.stack([src, dst], axis=0).astype(np.int64, copy=False)

    if make_undirected:
        reverse_edge_index = edge_index[[1, 0], :]
        edge_index = np.concatenate([edge_index, reverse_edge_index], axis=1)

        # Faster than np.unique(edge_index, axis=1) for large graphs.
        # It removes duplicate edges after making the graph undirected.
        edge_tuples = edge_index.T
        edge_tuples = np.ascontiguousarray(edge_tuples)
        edge_tuples = np.unique(edge_tuples, axis=0)
        edge_index = edge_tuples.T.astype(np.int64, copy=False)

    return edge_index