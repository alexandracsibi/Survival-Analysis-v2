from pathlib import Path
import json
from networkx import edges
import numpy as np


DATASET_NAMES = {
    "synthetic_binary",
    "synthetic_competing",
    "SEER",
    "MNB",
}


def load_feature_names(dataset_dir: Path, n_features: int):
    """Load feature names if available, otherwise create generic names."""
    path = dataset_dir / "feature_names.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            names = json.load(f)

        if len(names) != n_features:
            raise ValueError(
                f"feature_names.json has {len(names)} names, "
                f"but X has {n_features} features."
            )

        return names

    return [f"x{i}" for i in range(n_features)]


def load_split(dataset_dir: Path, split: str):
    """Load one split: train, val, or test."""
    X = np.load(dataset_dir / f"X_{split}.npy")
    time = np.load(dataset_dir / f"time_{split}.npy")
    event = np.load(dataset_dir / f"event_{split}.npy")

    if not (len(X) == len(time) == len(event)):
        raise ValueError(
            f"Length mismatch in {split}: "
            f"X={len(X)}, time={len(time)}, event={len(event)}"
        )

    return {
        "X": X.astype(np.float32),
        "time": time.astype(np.float32),
        "event": event.astype(np.int64),
    }

def make_time_bin_edges(train_time, n_time_bins: int, method: str = "quantile"):
    """
    Create time-bin edges from training times only.
    This avoids validation/test leakage.
    """
    train_time = np.asarray(train_time)

    if method == "quantile":
        quantiles = np.linspace(0, 1, n_time_bins + 1)
        edges = np.quantile(train_time, quantiles)
        edges = np.unique(edges)

        if len(edges) < 2:
            raise ValueError("Could not create valid quantile time bins.")

        return edges

    if method == "equal_width":
        return np.linspace(train_time.min(), train_time.max(), n_time_bins + 1)

    raise ValueError(f"Unknown time binning method: {method}")


def assign_time_bins(time, edges):
    """
    Assign each time value to a discrete bin index.
    Output range: 0 to n_bins - 1
    """
    time = np.asarray(time)

    # np.digitize returns 1..len(edges), so subtract 1
    bins = np.digitize(time, edges[1:-1], right=True)

    # safety clamp
    bins = np.clip(bins, 0, len(edges) - 2)

    return bins.astype(np.int64)


def add_time_bins(dataset, n_time_bins: int, method: str = "quantile"):
    """
    Add DeepHit time bins to train/val/test using train-time edges only.
    """
    edges = make_time_bin_edges(
        dataset["train"]["time"],
        n_time_bins=n_time_bins,
        method=method,
    )

    for split in ["train", "val", "test"]:
        dataset[split]["time_bin"] = assign_time_bins(
            dataset[split]["time"],
            edges,
        )

    requested_bins = n_time_bins
    actual_bins = len(edges) - 1

    if actual_bins < requested_bins:
        print(
            f"Warning: requested {requested_bins} time bins, "
            f"but only {actual_bins} unique bins were created."
        )

    dataset["time_bin_edges"] = edges
    dataset["n_time_bins"] = len(edges) - 1

    return dataset

def load_dataset(name: str, data_root: str | Path = "data"):
    """
    Load a prepared survival dataset in unified format.

    Expected files:
    - X_train.npy, X_val.npy, X_test.npy
    - time_train.npy, time_val.npy, time_test.npy
    - event_train.npy, event_val.npy, event_test.npy

    Optional:
    - time_bin_train.npy, time_bin_val.npy, time_bin_test.npy
    - feature_names.json
    """

    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset name: {name}. Available: {sorted(DATASET_NAMES)}")

    data_root = Path(data_root)
    dataset_dir = data_root / name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    train = load_split(dataset_dir, "train")
    val = load_split(dataset_dir, "val")
    test = load_split(dataset_dir, "test")

    n_features = train["X"].shape[1]
    feature_names = load_feature_names(dataset_dir, n_features)

    # Event convention:
    # 0 = censored
    # 1 = event for binary/single-risk
    # 2,3,... = competing event types
    all_events = np.concatenate([train["event"], val["event"], test["event"]])
    unique_events = sorted(np.unique(all_events).tolist())

    is_competing = any(e > 1 for e in unique_events)
    n_events = max(unique_events) if len(unique_events) > 0 else 1

    dataset = {
        "name": name,
        "path": dataset_dir,
        "train": train,
        "val": val,
        "test": test,
        "feature_names": feature_names,
        "n_features": n_features,
        "unique_events": unique_events,
        "is_competing": is_competing,
        "n_events": int(n_events),
    }

    return dataset


def print_dataset_summary(dataset):
    """Small readable summary for debugging."""
    print("=" * 80)
    print(f"Dataset: {dataset['name']}")
    print(f"Features: {dataset['n_features']}")
    print(f"Events: {dataset['unique_events']}")
    print(f"Competing risk: {dataset['is_competing']}")
    print(f"n_events: {dataset['n_events']}")

    for split in ["train", "val", "test"]:
        X = dataset[split]["X"]
        time = dataset[split]["time"]
        event = dataset[split]["event"]

        print(f"\n{split}:")
        print("  X:", X.shape)
        print("  time:", time.shape, "min:", time.min(), "max:", time.max())
        print("  event:", event.shape, dict(zip(*np.unique(event, return_counts=True))))