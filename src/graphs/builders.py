import numpy as np
from sklearn.neighbors import NearestNeighbors


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

    X = np.asarray(X)

    if k <= 0:
        raise ValueError("k must be positive.")

    n_nodes = X.shape[0]

    if k >= n_nodes:
        raise ValueError(f"k={k} must be smaller than number of nodes={n_nodes}.")

    # If we do not want self-loops, ask for k+1 neighbors because nearest neighbor is usually itself.
    n_neighbors = k if include_self else k + 1

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto",
        n_jobs=-1,
    )

    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    edges = []

    for src in range(n_nodes):
        for j, dst in enumerate(indices[src]):
            if not include_self and src == dst:
                continue

            edges.append((src, int(dst)))

    edge_index = np.array(edges, dtype=np.int64).T

    if make_undirected:
        reverse_edge_index = edge_index[[1, 0], :]
        edge_index = np.concatenate([edge_index, reverse_edge_index], axis=1)

        # Remove duplicate edges
        edge_index = np.unique(edge_index, axis=1)

    return edge_index