from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
import pandas as pd
import torch

from src.datasets.loaders import load_dataset
from src.models.baselines import DeepSurv, DeepHit
from src.training.baseline_runner import (
    predict_risk,
    predict_deephit_risk,
    predict_deephit_event_risk,
)
from src.evaluation import c_index, event_specific_c_index
from src.utils import get_device
from src.datasets.graph_dataset import make_graph_survival_data
from src.models.gnn_models import GraphSAGECoxModel, GraphSAGEDeepHitModel
from src.training.gnn_runner import (
    predict_gnn_deephit_risk,
    predict_gnn_deephit_event_risk,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    model_state_dict = torch.load(checkpoint_path, map_location="cpu")
    metadata_path = PROJECT_ROOT / "results" / "tables" / f"{checkpoint_path.stem}_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = metadata["config"]
    device = get_device(config["device"])

    ds = load_dataset(
        name=config["dataset"]["name"],
        data_root=config["dataset"]["data_root"],
    )

    model_name = config["model"]["name"]

    if model_name == "DeepSurv":
        model = DeepSurv(
            input_dim=metadata["n_features"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )

    elif model_name == "DeepHit":
        model = DeepHit(
            input_dim=metadata["n_features"],
            n_time_bins=metadata["n_time_bins"],
            n_events=ds["n_events"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GraphSAGECox":
        model = GraphSAGECoxModel(
            input_dim=metadata["n_features"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "GraphSAGEDeepHit":
        model = GraphSAGEDeepHitModel(
            input_dim=metadata["n_features"],
            n_time_bins=metadata["n_time_bins"],
            n_events=ds["n_events"],
            hidden_dims=config["model"]["hidden_dims"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    graph_data = None

    if model_name == "GraphSAGEDeepHit":
        from src.datasets.loaders import add_time_bins

        ds = add_time_bins(
            ds,
            n_time_bins=config["deephit"]["n_time_bins"],
            method=config["deephit"]["binning"],
        )

    if model_name in ["GraphSAGECox", "GraphSAGEDeepHit"]:
        edge_index_path = PROJECT_ROOT / config["graph"]["edge_index_path"]

        graph_data = make_graph_survival_data(
            dataset=ds,
            edge_index_path=edge_index_path,
        ).to(device)

    rows = []

    for split in ["train", "val", "test"]:
        if model_name == "DeepSurv":
            risk = predict_risk(
                model=model,
                X=ds[split]["X"],
                device=device,
            )

            event_binary = (ds[split]["event"] != 0).astype("int64")

            score = c_index(
                time=ds[split]["time"],
                event=event_binary,
                risk_score=risk,
            )

            rows.append({
                "split": split,
                "metric": "c_index_any_event",
                "event_id": "any",
                "value": score,
            })

            print(f"{split}: c_index_any_event={score:.4f}")

        elif model_name == "DeepHit":
            if ds["is_competing"]:
                event_ids = [e for e in ds["unique_events"] if e != 0]

                for event_id in event_ids:
                    risk = predict_deephit_event_risk(
                        model=model,
                        X=ds[split]["X"],
                        event_id=event_id,
                        device=device,
                    )

                    score = event_specific_c_index(
                        time=ds[split]["time"],
                        event=ds[split]["event"],
                        risk_score=risk,
                        event_id=event_id,
                    )

                    rows.append({
                        "split": split,
                        "metric": "event_specific_c_index",
                        "event_id": event_id,
                        "value": score,
                    })

                    print(
                        f"{split}: event_{event_id}_c_index={score:.4f}"
                    )

            else:
                risk = predict_deephit_risk(
                    model=model,
                    X=ds[split]["X"],
                    device=device,
                )

                event_binary = (ds[split]["event"] != 0).astype("int64")

                score = c_index(
                    time=ds[split]["time"],
                    event=event_binary,
                    risk_score=risk,
                )

                rows.append({
                    "split": split,
                    "metric": "c_index",
                    "event_id": 1,
                    "value": score,
                })

        elif model_name == "GraphSAGECox":
            model.eval()

            with torch.no_grad():
                risk_all = model(
                    graph_data.x,
                    graph_data.edge_index,
                )

            mask = getattr(graph_data, f"{split}_mask")

            risk = risk_all[mask].detach().cpu().numpy()
            time = graph_data.time[mask].detach().cpu().numpy()
            event = graph_data.event[mask].detach().cpu().numpy()
            event_binary = (event != 0).astype("int64")

            score = c_index(
                time=time,
                event=event_binary,
                risk_score=risk,
            )

            rows.append({
                "split": split,
                "metric": "c_index_any_event",
                "event_id": "any",
                "value": score,
            })

        elif model_name == "GraphSAGEDeepHit":
            mask = getattr(graph_data, f"{split}_mask")

            event = graph_data.event[mask].detach().cpu().numpy()
            time = graph_data.time[mask].detach().cpu().numpy()

            if ds["is_competing"]:
                event_ids = [e for e in ds["unique_events"] if e != 0]

                for event_id in event_ids:
                    risk = predict_gnn_deephit_event_risk(
                        model=model,
                        data=graph_data,
                        mask=mask,
                        event_id=event_id,
                        device=device,
                    )

                    score = event_specific_c_index(
                        time=time,
                        event=event,
                        risk_score=risk,
                        event_id=event_id,
                    )

                    rows.append({
                        "split": split,
                        "metric": "event_specific_c_index",
                        "event_id": event_id,
                        "value": score,
                    })

                    print(f"{split}: event_{event_id}_c_index={score:.4f}")

            else:
                risk = predict_gnn_deephit_risk(
                    model=model,
                    data=graph_data,
                    mask=mask,
                    device=device,
                )

                event_binary = (event != 0).astype("int64")

                score = c_index(
                    time=time,
                    event=event_binary,
                    risk_score=risk,
                )

                rows.append({
                    "split": split,
                    "metric": "c_index",
                    "event_id": 1,
                    "value": score,
                })

                print(f"{split}: c_index={score:.4f}")

    results_dir = PROJECT_ROOT / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{checkpoint_path.stem}_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()