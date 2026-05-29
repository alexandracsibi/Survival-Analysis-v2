from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json

import pandas as pd
import torch

from src.models.baselines import DeepSurv, DeepHit
from src.training.baseline_runner import train_deepsurv, train_deephit
from src.datasets.loaders import load_dataset, add_time_bins
from src.utils import load_config, set_seed, get_device
from src.datasets.graph_dataset import make_graph_survival_data
from src.models.gnn_models import GraphSAGECoxModel, GraphSAGEDeepHitModel
from src.training.gnn_runner import (
    train_gnn_cox,
    train_gnn_cox_sampled,
    train_gnn_deephit,
    train_gnn_deephit_sampled,
)
from src.training.ssl_runner import (
    train_gnn_deephit_ssl_static_teacher,
    train_gnn_deephit_ssl_static_teacher_sampled,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["seed"])
    device = get_device(config["device"])

    print("Using device:", device)

    ds = load_dataset(
        name=config["dataset"]["name"],
        data_root=config["dataset"]["data_root"],
    )

    model_name = config["model"]["name"]

    if model_name == "DeepSurv":
        model = DeepSurv(
            input_dim=ds["n_features"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )

        model, history = train_deepsurv(
            model=model,
            train_data=ds["train"],
            val_data=ds["val"],
            device=device,
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=config["training"]["epochs"],
            early_stopping=config.get("early_stopping"),
        )

    elif model_name == "DeepHit":
        event_mode = config.get("dataset", {}).get("event_mode")

        if event_mode == "binary_any_event":
            for split in ["train", "val", "test"]:
                if split in ds:
                    ds[split]["event"] = (ds[split]["event"] != 0).astype("int64")

            ds["n_events"] = 1

        ds = add_time_bins(
            ds,
            n_time_bins=config["deephit"]["n_time_bins"],
            method=config["deephit"]["binning"],
        )

        model = DeepHit(
            input_dim=ds["n_features"],
            n_time_bins=ds["n_time_bins"],
            n_events=ds["n_events"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )

        model, history = train_deephit(
            model=model,
            train_data=ds["train"],
            val_data=ds["val"],
            device=device,
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            epochs=config["training"]["epochs"],
            alpha=config["deephit"]["alpha"],
            beta=config["deephit"]["beta"],
            sigma=config["deephit"]["sigma"],
            max_rank_pairs=config["deephit"].get("max_rank_pairs"),
            early_stopping=config.get("early_stopping"),
        )
    elif model_name == "GraphSAGECox":
        edge_index_path = PROJECT_ROOT / config["graph"]["edge_index_path"]

        graph_data = make_graph_survival_data(
            dataset=ds,
            edge_index_path=edge_index_path,
        )

        model = GraphSAGECoxModel(
            input_dim=ds["n_features"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )

        sampling_config = config.get("sampling", {})
        use_sampling = sampling_config.get("enabled", False)

        if use_sampling:
            model, history = train_gnn_cox_sampled(
                model=model,
                data=graph_data,
                device=device,
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                epochs=config["training"]["epochs"],
                batch_size=sampling_config.get("batch_size", 4096),
                num_neighbors=sampling_config.get("num_neighbors", [5, 5]),
                early_stopping=config.get("early_stopping"),
                semi_supervised=config.get("semi_supervised"),
            )
        else:
            model, history = train_gnn_cox(
                model=model,
                data=graph_data,
                device=device,
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                epochs=config["training"]["epochs"],
                early_stopping=config.get("early_stopping"),
            )

    elif model_name == "GraphSAGEDeepHit":
        ds = add_time_bins(
            ds,
            n_time_bins=config["deephit"]["n_time_bins"],
            method=config["deephit"]["binning"],
        )

        edge_index_path = PROJECT_ROOT / config["graph"]["edge_index_path"]

        graph_data = make_graph_survival_data(
            dataset=ds,
            edge_index_path=edge_index_path,
        )

        model = GraphSAGEDeepHitModel(
            input_dim=ds["n_features"],
            n_time_bins=ds["n_time_bins"],
            n_events=ds["n_events"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )

        if config.get("ssl", {}).get("enabled", False):
            sampling_config = config.get("sampling", {})
            use_sampling = sampling_config.get("enabled", False)

            teacher_checkpoint = config["ssl"].get("teacher_checkpoint")

            teacher_mode = config["ssl"].get("teacher_mode", "static")
            pseudo_label_update = config["ssl"].get("pseudo_label_update", "once")
            enforce_censoring_consistency = config["ssl"].get(
                "enforce_censoring_consistency",
                True,
            )
            candidate = config["ssl"].get("candidate", "censored_train")

            if teacher_checkpoint is not None and config["ssl"].get("init_student_from_teacher", True):
                teacher_checkpoint_path = PROJECT_ROOT / teacher_checkpoint
                state_dict = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
            else:
                teacher_checkpoint_path = PROJECT_ROOT / teacher_checkpoint if teacher_checkpoint is not None else None

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
            )

            if use_sampling:
                model, history, pseudo = train_gnn_deephit_ssl_static_teacher_sampled(
                    model=model,
                    data=graph_data,
                    optimizer=optimizer,
                    n_epochs=config["training"]["epochs"],
                    batch_size=sampling_config.get("batch_size", 4096),
                    num_neighbors=sampling_config.get("num_neighbors", [5, 5]),
                    alpha=config["deephit"]["alpha"],
                    beta=config["deephit"]["beta"],
                    sigma=config["deephit"]["sigma"],
                    pseudo_weight=config["ssl"]["pseudo_weight"],
                    min_confidence=config["ssl"]["min_confidence"],
                    teacher_checkpoint_path=teacher_checkpoint_path,
                    teacher_mode=teacher_mode,
                    pseudo_label_update=pseudo_label_update,
                    enforce_censoring_consistency=enforce_censoring_consistency,
                    candidate=candidate,
                    semi_supervised=config.get("semi_supervised"),
                    patience=config["early_stopping"]["patience"],
                    min_delta=config["early_stopping"]["min_delta"],
                    device=device,
                )
            else:
                model, history, pseudo = train_gnn_deephit_ssl_static_teacher(
                    model=model,
                    data=graph_data,
                    optimizer=optimizer,
                    n_epochs=config["training"]["epochs"],
                    alpha=config["deephit"]["alpha"],
                    beta=config["deephit"]["beta"],
                    sigma=config["deephit"]["sigma"],
                    pseudo_weight=config["ssl"]["pseudo_weight"],
                    min_confidence=config["ssl"]["min_confidence"],
                    teacher_checkpoint_path=teacher_checkpoint_path,
                    teacher_mode=teacher_mode,
                    pseudo_label_update=pseudo_label_update,
                    enforce_censoring_consistency=enforce_censoring_consistency,
                    candidate=candidate,
                    patience=config["early_stopping"]["patience"],
                    min_delta=config["early_stopping"]["min_delta"],
                    device=device,
                )
        else:
            sampling_config = config.get("sampling", {})
            use_sampling = sampling_config.get("enabled", False)

            if use_sampling:
                model, history = train_gnn_deephit_sampled(
                    model=model,
                    data=graph_data,
                    device=device,
                    lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"],
                    epochs=config["training"]["epochs"],
                    batch_size=sampling_config.get("batch_size", 4096),
                    num_neighbors=sampling_config.get("num_neighbors", [10, 10]),
                    alpha=config["deephit"]["alpha"],
                    beta=config["deephit"]["beta"],
                    sigma=config["deephit"]["sigma"],
                    early_stopping=config.get("early_stopping"),
                    semi_supervised=config.get("semi_supervised"),
                )
            else:
                model, history = train_gnn_deephit(
                    model=model,
                    data=graph_data,
                    device=device,
                    lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"],
                    epochs=config["training"]["epochs"],
                    alpha=config["deephit"]["alpha"],
                    beta=config["deephit"]["beta"],
                    sigma=config["deephit"]["sigma"],
                    early_stopping=config.get("early_stopping"),
                )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    config_path = Path(args.config)
    relative_config_path = config_path.with_suffix("").relative_to("configs")

    experiment_name = relative_config_path.name
    experiment_group = relative_config_path.parent

    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir = results_dir / "checkpoints" / experiment_group
    table_dir = results_dir / "tables" / experiment_group

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Save training history
    history_path = table_dir / f"{experiment_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Save model checkpoint
    checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    graph_metadata_path = None

    if config.get("graph") is not None:
        edge_index_path = Path(config["graph"]["edge_index_path"])
        graph_metadata_path = str(
            edge_index_path.with_name(
                edge_index_path.name.replace("_edge_index.npy", "_metadata.json")
            )
        )

    ssl_enabled = config.get("ssl", {}).get("enabled", False)

    pseudo_label_stats = None
    selected_pseudo = None

    if ssl_enabled:
        selected_pseudo = int(pseudo.selected_mask.sum().item())

        pseudo_label_stats = {
            "selected_pseudo": selected_pseudo,
            "total_nodes": int(pseudo.selected_mask.shape[0]),
            "selection_rate": float(pseudo.selected_mask.float().mean().item()),
            "mean_confidence": float(pseudo.confidence.mean().item()),
            "max_confidence": float(pseudo.confidence.max().item()),
            "min_confidence": float(pseudo.confidence.min().item()),
        }

    metadata_path = table_dir / f"{experiment_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "n_features": ds["n_features"],
            "feature_names": ds["feature_names"],
            "model_name": model_name,

            "graph": config.get("graph"),
            "graph_metadata_path": graph_metadata_path,

            "ssl": config.get("ssl"),
            "selected_pseudo": selected_pseudo,
            "pseudo_label_stats": pseudo_label_stats,

            "n_time_bins": ds.get("n_time_bins"),
            "time_bin_edges": (
                ds.get("time_bin_edges").tolist()
                if ds.get("time_bin_edges") is not None
                else None
            ),
        }, f, indent=2)

    # Save config copy
    config_path = table_dir / f"{experiment_name}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nSaved:")
    print(f"- {history_path}")
    print(f"- {checkpoint_path}")
    print(f"- {config_path}")
    print(f"- {metadata_path}")


if __name__ == "__main__":
    main()