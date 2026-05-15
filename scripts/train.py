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
from src.training.gnn_runner import train_gnn_cox, train_gnn_deephit
from src.training.ssl_runner import train_gnn_deephit_ssl_static_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config["seed"])
    device = get_device(config["device"])

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
            model, history, pseudo = train_gnn_deephit_ssl_static_teacher(
                model=model,
                data=graph_data,
                optimizer=torch.optim.Adam(
                    model.parameters(),
                    lr=config["training"]["lr"],
                    weight_decay=config["training"]["weight_decay"],
                ),
                n_epochs=config["training"]["epochs"],
                alpha=config["deephit"]["alpha"],
                beta=config["deephit"]["beta"],
                sigma=config["deephit"]["sigma"],
                pseudo_weight=config["ssl"]["pseudo_weight"],
                min_confidence=config["ssl"]["min_confidence"],
                patience=config["early_stopping"]["patience"],
                min_delta=config["early_stopping"]["min_delta"],
                device=device,
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

    experiment_name = Path(args.config).stem

    results_dir = PROJECT_ROOT / "results"
    checkpoint_dir = results_dir / "checkpoints"
    table_dir = results_dir / "tables"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Save training history
    history_path = table_dir / f"{experiment_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Save model checkpoint
    checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metadata_path = table_dir / f"{experiment_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "n_features": ds["n_features"],
            "feature_names": ds["feature_names"],
            "model_name": model_name,
            "graph": config.get("graph"),
            "ssl": config.get("ssl"),
            "selected_pseudo": int(pseudo.selected_mask.sum().item()) if config.get("ssl", {}).get("enabled", False) else None,
            "n_time_bins": ds.get("n_time_bins"),
            "time_bin_edges": ds.get("time_bin_edges").tolist() if ds.get("time_bin_edges") is not None else None,
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