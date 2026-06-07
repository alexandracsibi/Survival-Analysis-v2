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
from src.training.baseline_runner import predict_risk
from src.evaluation import (
    c_index,
    event_specific_c_index,
    time_dependent_auc,
    choose_eval_times,
    deephit_survival_at_times,
    brier_and_ibs,
    deephit_risk_scores,
)
from src.utils import get_device
from src.datasets.graph_dataset import make_graph_survival_data
from src.models.gnn_models import GraphSAGECoxModel, GraphSAGEDeepHitModel


def logits_to_deephit_probs(logits):
    """
    Convert DeepHit logits [N, K, T] to joint event-time probabilities [N, K, T].
    """
    n, k, t = logits.shape
    probs = torch.softmax(logits.reshape(n, k * t), dim=1).reshape(n, k, t)
    return probs


def append_risk_variant_metrics(
    rows,
    split,
    time,
    event_binary,
    train_time,
    train_event,
    risk_variants,
    event_id,
    compatibility_prefix=None,
):
    """
    Add C-index and time-dependent AUC for all DeepHit risk-score variants.

    risk_variants:
        dict from deephit_risk_scores()

    event_id:
        "any" for binary / any-event evaluation,
        or original event label for competing-risk evaluation.

    compatibility_prefix:
        None for binary metrics.
        "event_specific" for competing-risk backward-compatible names.
    """
    main_score = None
    main_mean_auc = None

    for risk_method, risk in risk_variants.items():
        score = c_index(
            time=time,
            event=event_binary,
            risk_score=risk,
        )

        auc_by_time, mean_auc = time_dependent_auc(
            train_time=train_time,
            train_event=train_event,
            test_time=time,
            test_event=event_binary,
            risk_score=risk,
            n_times=5,
        )

        if compatibility_prefix == "event_specific":
            cindex_metric = f"event_specific_c_index_{risk_method}"
        else:
            cindex_metric = f"c_index_{risk_method}"

        rows.append({
            "split": split,
            "metric": cindex_metric,
            "event_id": event_id,
            "value": score,
        })

        rows.append({
            "split": split,
            "metric": f"td_auc_mean_{risk_method}",
            "event_id": event_id,
            "value": mean_auc,
        })

        for eval_time, auc_value in auc_by_time.items():
            rows.append({
                "split": split,
                "metric": f"td_auc_at_{eval_time:.2f}_{risk_method}",
                "event_id": event_id,
                "value": auc_value,
            })

        if risk_method == "expected_time":
            main_score = score
            main_mean_auc = mean_auc

            if compatibility_prefix == "event_specific":
                rows.append({
                    "split": split,
                    "metric": "event_specific_c_index",
                    "event_id": event_id,
                    "value": score,
                })
            else:
                rows.append({
                    "split": split,
                    "metric": "c_index",
                    "event_id": event_id,
                    "value": score,
                })

            rows.append({
                "split": split,
                "metric": "td_auc_mean",
                "event_id": event_id,
                "value": mean_auc,
            })

            for eval_time, auc_value in auc_by_time.items():
                rows.append({
                    "split": split,
                    "metric": f"td_auc_at_{eval_time:.2f}",
                    "event_id": event_id,
                    "value": auc_value,
                })

    return main_score, main_mean_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_relative = checkpoint_path.relative_to(
        PROJECT_ROOT / "results" / "checkpoints"
    )

    model_state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    metadata_path = (
        PROJECT_ROOT / "results" / "tables" / checkpoint_relative.with_suffix("")
    ).with_name(f"{checkpoint_path.stem}_metadata.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = metadata["config"]
    device = get_device(config["device"])

    ds = load_dataset(
        name=config["dataset"]["name"],
        data_root=config["dataset"]["data_root"],
    )

    model_name = config["model"]["name"]
    event_mode = config.get("dataset", {}).get("event_mode")

    if event_mode == "binary_any_event":
        for split in ["train", "val", "test"]:
            if split in ds:
                ds[split]["event"] = (ds[split]["event"] != 0).astype("int64")

        ds["n_events"] = 1
        ds["is_competing"] = False
        ds["unique_events"] = [0, 1]

    if model_name == "DeepHit" or model_name == "GraphSAGEDeepHit":
        from src.datasets.loaders import add_time_bins

        ds = add_time_bins(
            ds,
            n_time_bins=config["deephit"]["n_time_bins"],
            method=config["deephit"]["binning"],
        )

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
    model.eval()

    graph_data = None

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
            time = ds[split]["time"]

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

            train_time = ds["train"]["time"]
            train_event = (ds["train"]["event"] != 0).astype("int64")

            auc_by_time, mean_auc = time_dependent_auc(
                train_time=train_time,
                train_event=train_event,
                test_time=time,
                test_event=event_binary,
                risk_score=risk,
                n_times=5,
            )

            rows.append({
                "split": split,
                "metric": "td_auc_mean",
                "event_id": "any",
                "value": mean_auc,
            })

            for eval_time, auc_value in auc_by_time.items():
                rows.append({
                    "split": split,
                    "metric": f"td_auc_at_{eval_time:.2f}",
                    "event_id": "any",
                    "value": auc_value,
                })

            print(
                f"{split}: "
                f"c_index_any_event={score:.4f}, "
                f"td_auc_mean={mean_auc:.4f}"
            )

        elif model_name == "DeepHit":
            train_time = ds["train"]["time"]

            with torch.no_grad():
                X_tensor = torch.tensor(
                    ds[split]["X"],
                    dtype=torch.float32,
                ).to(device)

                logits = model(X_tensor)
                probs_tensor = logits_to_deephit_probs(logits)
                probs = probs_tensor.detach().cpu().numpy()

            if ds["is_competing"]:
                event_ids = [e for e in ds["unique_events"] if e != 0]

                for event_id in event_ids:
                    event_specific = (ds[split]["event"] == event_id).astype("int64")
                    train_event_specific = (
                        ds["train"]["event"] == event_id
                    ).astype("int64")

                    risk_variants = deephit_risk_scores(
                        probs=probs,
                        event_id=event_id,
                    )

                    main_score, main_mean_auc = None, None

                    for risk_method, risk in risk_variants.items():
                        score = event_specific_c_index(
                            time=ds[split]["time"],
                            event=ds[split]["event"],
                            risk_score=risk,
                            event_id=event_id,
                        )

                        auc_by_time, mean_auc = time_dependent_auc(
                            train_time=train_time,
                            train_event=train_event_specific,
                            test_time=ds[split]["time"],
                            test_event=event_specific,
                            risk_score=risk,
                            n_times=5,
                        )

                        rows.append({
                            "split": split,
                            "metric": f"event_specific_c_index_{risk_method}",
                            "event_id": event_id,
                            "value": score,
                        })

                        rows.append({
                            "split": split,
                            "metric": f"td_auc_mean_{risk_method}",
                            "event_id": event_id,
                            "value": mean_auc,
                        })

                        for eval_time, auc_value in auc_by_time.items():
                            rows.append({
                                "split": split,
                                "metric": f"td_auc_at_{eval_time:.2f}_{risk_method}",
                                "event_id": event_id,
                                "value": auc_value,
                            })

                        if risk_method == "expected_time":
                            main_score = score
                            main_mean_auc = mean_auc

                            rows.append({
                                "split": split,
                                "metric": "event_specific_c_index",
                                "event_id": event_id,
                                "value": score,
                            })

                            rows.append({
                                "split": split,
                                "metric": "td_auc_mean",
                                "event_id": event_id,
                                "value": mean_auc,
                            })

                            for eval_time, auc_value in auc_by_time.items():
                                rows.append({
                                    "split": split,
                                    "metric": f"td_auc_at_{eval_time:.2f}",
                                    "event_id": event_id,
                                    "value": auc_value,
                                })

                    print(
                        f"{split}: "
                        f"event_{event_id}_c_index={main_score:.4f}, "
                        f"event_{event_id}_td_auc_mean={main_mean_auc:.4f}"
                    )

            else:
                event_binary = (ds[split]["event"] != 0).astype("int64")
                time = ds[split]["time"]

                train_event = (ds["train"]["event"] != 0).astype("int64")

                risk_variants = deephit_risk_scores(
                    probs=probs,
                    event_id="any",
                )

                main_score, main_mean_auc = append_risk_variant_metrics(
                    rows=rows,
                    split=split,
                    time=time,
                    event_binary=event_binary,
                    train_time=train_time,
                    train_event=train_event,
                    risk_variants=risk_variants,
                    event_id="any",
                )

                eval_times = choose_eval_times(
                    train_time=train_time,
                    train_event=train_event,
                    n_times=5,
                )

                eval_times = eval_times[
                    (eval_times > time.min()) &
                    (eval_times < time.max())
                ]

                ibs = float("nan")

                if len(eval_times) > 1:
                    survival_probs = deephit_survival_at_times(
                        probs=probs,
                        time_bin_edges=ds["time_bin_edges"],
                        eval_times=eval_times,
                    )

                    brier_by_time, ibs = brier_and_ibs(
                        train_time=train_time,
                        train_event=train_event,
                        test_time=time,
                        test_event=event_binary,
                        survival_probs=survival_probs,
                        eval_times=eval_times,
                    )

                    rows.append({
                        "split": split,
                        "metric": "ibs",
                        "event_id": "any",
                        "value": ibs,
                    })

                    for eval_time, brier_value in brier_by_time.items():
                        rows.append({
                            "split": split,
                            "metric": f"brier_at_{eval_time:.2f}",
                            "event_id": "any",
                            "value": brier_value,
                        })

                print(
                    f"{split}: "
                    f"c_index_any_event={main_score:.4f}, "
                    f"td_auc_mean={main_mean_auc:.4f}, "
                    f"ibs={ibs:.4f}"
                )

        elif model_name == "GraphSAGECox":
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

            train_time = ds["train"]["time"]
            train_event = (ds["train"]["event"] != 0).astype("int64")

            auc_by_time, mean_auc = time_dependent_auc(
                train_time=train_time,
                train_event=train_event,
                test_time=time,
                test_event=event_binary,
                risk_score=risk,
                n_times=5,
            )

            rows.append({
                "split": split,
                "metric": "td_auc_mean",
                "event_id": "any",
                "value": mean_auc,
            })

            for eval_time, auc_value in auc_by_time.items():
                rows.append({
                    "split": split,
                    "metric": f"td_auc_at_{eval_time:.2f}",
                    "event_id": "any",
                    "value": auc_value,
                })

            print(
                f"{split}: "
                f"c_index_any_event={score:.4f}, "
                f"td_auc_mean={mean_auc:.4f}"
            )

        elif model_name == "GraphSAGEDeepHit":
            mask = getattr(graph_data, f"{split}_mask")

            event = graph_data.event[mask].detach().cpu().numpy()
            time = graph_data.time[mask].detach().cpu().numpy()

            with torch.no_grad():
                logits = model(graph_data.x, graph_data.edge_index)
                probs_all = logits_to_deephit_probs(logits).detach().cpu().numpy()

            mask_np = mask.detach().cpu().numpy()
            probs = probs_all[mask_np]

            train_time = ds["train"]["time"]

            if ds["is_competing"]:
                event_ids = [e for e in ds["unique_events"] if e != 0]

                for event_id in event_ids:
                    event_specific = (event == event_id).astype("int64")
                    train_event_specific = (
                        ds["train"]["event"] == event_id
                    ).astype("int64")

                    risk_variants = deephit_risk_scores(
                        probs=probs,
                        event_id=event_id,
                    )

                    main_score, main_mean_auc = None, None

                    for risk_method, risk in risk_variants.items():
                        score = event_specific_c_index(
                            time=time,
                            event=event,
                            risk_score=risk,
                            event_id=event_id,
                        )

                        auc_by_time, mean_auc = time_dependent_auc(
                            train_time=train_time,
                            train_event=train_event_specific,
                            test_time=time,
                            test_event=event_specific,
                            risk_score=risk,
                            n_times=5,
                        )

                        rows.append({
                            "split": split,
                            "metric": f"event_specific_c_index_{risk_method}",
                            "event_id": event_id,
                            "value": score,
                        })

                        rows.append({
                            "split": split,
                            "metric": f"td_auc_mean_{risk_method}",
                            "event_id": event_id,
                            "value": mean_auc,
                        })

                        for eval_time, auc_value in auc_by_time.items():
                            rows.append({
                                "split": split,
                                "metric": f"td_auc_at_{eval_time:.2f}_{risk_method}",
                                "event_id": event_id,
                                "value": auc_value,
                            })

                        if risk_method == "expected_time":
                            main_score = score
                            main_mean_auc = mean_auc

                            rows.append({
                                "split": split,
                                "metric": "event_specific_c_index",
                                "event_id": event_id,
                                "value": score,
                            })

                            rows.append({
                                "split": split,
                                "metric": "td_auc_mean",
                                "event_id": event_id,
                                "value": mean_auc,
                            })

                            for eval_time, auc_value in auc_by_time.items():
                                rows.append({
                                    "split": split,
                                    "metric": f"td_auc_at_{eval_time:.2f}",
                                    "event_id": event_id,
                                    "value": auc_value,
                                })

                    print(
                        f"{split}: "
                        f"event_{event_id}_c_index={main_score:.4f}, "
                        f"event_{event_id}_td_auc_mean={main_mean_auc:.4f}"
                    )

            else:
                event_binary = (event != 0).astype("int64")
                train_event = (ds["train"]["event"] != 0).astype("int64")

                risk_variants = deephit_risk_scores(
                    probs=probs,
                    event_id="any",
                )

                main_score, main_mean_auc = append_risk_variant_metrics(
                    rows=rows,
                    split=split,
                    time=time,
                    event_binary=event_binary,
                    train_time=train_time,
                    train_event=train_event,
                    risk_variants=risk_variants,
                    event_id="any",
                )

                eval_times = choose_eval_times(
                    train_time=train_time,
                    train_event=train_event,
                    n_times=5,
                )

                eval_times = eval_times[
                    (eval_times > time.min()) &
                    (eval_times < time.max())
                ]

                ibs = float("nan")

                if len(eval_times) > 1:
                    survival_probs = deephit_survival_at_times(
                        probs=probs,
                        time_bin_edges=ds["time_bin_edges"],
                        eval_times=eval_times,
                    )

                    brier_by_time, ibs = brier_and_ibs(
                        train_time=train_time,
                        train_event=train_event,
                        test_time=time,
                        test_event=event_binary,
                        survival_probs=survival_probs,
                        eval_times=eval_times,
                    )

                    rows.append({
                        "split": split,
                        "metric": "ibs",
                        "event_id": "any",
                        "value": ibs,
                    })

                    for eval_time, brier_value in brier_by_time.items():
                        rows.append({
                            "split": split,
                            "metric": f"brier_at_{eval_time:.2f}",
                            "event_id": "any",
                            "value": brier_value,
                        })

                print(
                    f"{split}: "
                    f"c_index_any_event={main_score:.4f}, "
                    f"td_auc_mean={main_mean_auc:.4f}, "
                    f"ibs={ibs:.4f}"
                )

    results_dir = metadata_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{checkpoint_path.stem}_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()