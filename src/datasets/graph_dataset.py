from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data


def load_edge_index(path):
    edge_index = np.load(path)
    return torch.tensor(edge_index, dtype=torch.long)


def make_split_masks(n_train, n_val, n_test):
    n_total = n_train + n_val + n_test

    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)

    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    return train_mask, val_mask, test_mask


def make_graph_survival_data(dataset, edge_index_path):
    """
    Package survival dataset + graph into PyG Data object.

    Node order:
    train nodes first, then val nodes, then test nodes.
    """

    X_all = np.concatenate(
        [
            dataset["train"]["X"],
            dataset["val"]["X"],
            dataset["test"]["X"],
        ],
        axis=0,
    )

    time_all = np.concatenate(
        [
            dataset["train"]["time"],
            dataset["val"]["time"],
            dataset["test"]["time"],
        ],
        axis=0,
    )

    event_all = np.concatenate(
        [
            dataset["train"]["event"],
            dataset["val"]["event"],
            dataset["test"]["event"],
        ],
        axis=0,
    )
    time_bin_all = None

    if "time_bin" in dataset["train"]:
        time_bin_all = np.concatenate(
            [
                dataset["train"]["time_bin"],
                dataset["val"]["time_bin"],
                dataset["test"]["time_bin"],
            ],
            axis=0,
        )

    n_train = len(dataset["train"]["X"])
    n_val = len(dataset["val"]["X"])
    n_test = len(dataset["test"]["X"])

    train_mask, val_mask, test_mask = make_split_masks(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    edge_index = load_edge_index(edge_index_path)

    data = Data(
        x=torch.tensor(X_all, dtype=torch.float32),
        edge_index=edge_index,
        time=torch.tensor(time_all, dtype=torch.float32),
        event=torch.tensor(event_all, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    if time_bin_all is not None:
        data.time_bin = torch.tensor(time_bin_all, dtype=torch.long)

    return data