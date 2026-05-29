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

    n_neighbors = k if include_self else k + 1

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto",
        n_jobs=-1,
    )
    nbrs.fit(X)

    _, indices = nbrs.kneighbors(X)

    src = np.repeat(np.arange(n_nodes), n_neighbors)
    dst = indices.reshape(-1)

    if not include_self:
        keep = src != dst
        src = src[keep]
        dst = dst[keep]

        # keep exactly first k non-self neighbors per node
        src = src.reshape(n_nodes, -1)[:, :k].reshape(-1)
        dst = dst.reshape(n_nodes, -1)[:, :k].reshape(-1)

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    if make_undirected:
        reverse_edge_index = edge_index[[1, 0], :]
        edge_index = np.concatenate([edge_index, reverse_edge_index], axis=1)
        edge_index = np.unique(edge_index, axis=1)

    return edge_index