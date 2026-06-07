from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
import numpy as np

from src.datasets.loaders import load_dataset
from src.graphs.builders import build_knn_graph
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_name = config["dataset"]["name"]
    data_root = config["dataset"]["data_root"]

    graph_config = config.get("graph", {})
    k = graph_config.get("k", 10)
    metric = graph_config.get("metric", "euclidean")
    include_self = graph_config.get("include_self", False)
    make_undirected = graph_config.get("make_undirected", True)
    batch_size = graph_config.get("batch_size", 100_000)

    ds = load_dataset(dataset_name, data_root=data_root)

    X_all = np.concatenate(
        [
            ds["train"]["X"],
            ds["val"]["X"],
            ds["test"]["X"],
        ],
        axis=0,
    )

    n_train = len(ds["train"]["X"])
    n_val = len(ds["val"]["X"])
    n_test = len(ds["test"]["X"])

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    print("Building graph...")
    print("Dataset:", dataset_name)
    print("X_all:", X_all.shape)
    print("k:", k)
    print("metric:", metric)

    edge_index = build_knn_graph(
        X_all,
        k=k,
        metric=metric,
        include_self=include_self,
        make_undirected=make_undirected,
        batch_size=batch_size,
    )

    out_dir = PROJECT_ROOT / "data" / dataset_name / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_name = f"knn_k{k}_{metric}"
    edge_path = out_dir / f"{graph_name}_edge_index.npy"
    meta_path = out_dir / f"{graph_name}_metadata.json"

    np.save(edge_path, edge_index)

    feature_names = ds.get("feature_names", None)

    preprocessing_note = graph_config.get(
        "preprocessing_note",
        "Graph is built from the prepared X_train/X_val/X_test arrays loaded by load_dataset. "
        "These arrays are assumed to already contain the final preprocessed/scaled feature representation."
    )

    metadata = {
        "dataset": dataset_name,
        "graph_name": graph_name,

        "graph_type": "knn_feature_similarity",
        "k": k,
        "metric": metric,
        "include_self": include_self,
        "make_undirected": make_undirected,

        "transductive": True,
        "transductive_note": (
            "Graph nodes include train, validation, and test samples. "
            "Only input features X are used for graph construction; labels, event times, "
            "outcomes, and model predictions are not used."
        ),

        "preprocessing_note": preprocessing_note,
        "feature_count": int(X_all.shape[1]),
        "feature_names": feature_names,

        "included_feature_groups": graph_config.get("included_feature_groups"),
        "dropped_feature_groups": graph_config.get("dropped_feature_groups"),

        "n_nodes": int(X_all.shape[0]),
        "n_edges": int(edge_index.shape[1]),

        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),

        "node_order": "train_then_val_then_test",

        "train_idx_start": int(train_idx[0]),
        "train_idx_end": int(train_idx[-1]),
        "val_idx_start": int(val_idx[0]),
        "val_idx_end": int(val_idx[-1]),
        "test_idx_start": int(test_idx[0]),
        "test_idx_end": int(test_idx[-1]),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:")
    print("-", edge_path)
    print("-", meta_path)


if __name__ == "__main__":
    main()