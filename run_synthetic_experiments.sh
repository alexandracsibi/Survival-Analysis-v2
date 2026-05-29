#!/usr/bin/env bash

mkdir -p results/logs
mkdir -p results/figures/auto

FAILED_FILE="results/logs/failed_experiments.txt"
: > "$FAILED_FILE"

run_cmd () {
    DESC="$1"
    CMD="$2"
    LOG="$3"

    echo "============================================================"
    echo "$DESC"
    echo "Command: $CMD"
    echo "Log: $LOG"
    echo "============================================================"

    bash -c "$CMD" 2>&1 | tee "$LOG"
    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED: $DESC" | tee -a "$FAILED_FILE"
        echo "Command: $CMD" >> "$FAILED_FILE"
        echo "Log: $LOG" >> "$FAILED_FILE"
        echo "Exit code: $STATUS" >> "$FAILED_FILE"
        echo "" >> "$FAILED_FILE"
        echo "Skipping to next step..."
        return 0
    fi

    echo "Finished: $DESC"
    echo
    return 0
}

run_exp () {
    CONFIG="$1"

    if [ ! -f "$CONFIG" ]; then
        echo "SKIP: config does not exist: $CONFIG" | tee -a "$FAILED_FILE"
        return 0
    fi

    NAME=$(echo "$CONFIG" | sed 's#configs/##' | sed 's#/#_#g' | sed 's#.yaml##')
    CHECKPOINT=$(echo "$CONFIG" | sed 's#configs#results/checkpoints#' | sed 's#.yaml#.pt#')

    run_cmd \
        "Training $CONFIG" \
        "python scripts/train.py --config $CONFIG" \
        "results/logs/${NAME}_train.log"

    if [ -f "$CHECKPOINT" ]; then
        run_cmd \
            "Evaluating $CHECKPOINT" \
            "python scripts/evaluate.py --checkpoint $CHECKPOINT" \
            "results/logs/${NAME}_eval.log"
    else
        echo "SKIP evaluation: checkpoint not found: $CHECKPOINT" | tee -a "$FAILED_FILE"
    fi
}

build_graph_if_missing () {
    DESC="$1"
    CONFIG="$2"
    GRAPH_PATH="$3"
    LOG="$4"

    if [ -f "$GRAPH_PATH" ]; then
        echo "SKIP: $DESC"
        echo "Graph already exists: $GRAPH_PATH"
        echo
        return 0
    fi

    run_cmd \
        "$DESC" \
        "python scripts/build_graphs.py --config $CONFIG" \
        "$LOG"
}

plot_histories () {
python - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out_dir = Path("results/figures/auto")
out_dir.mkdir(parents=True, exist_ok=True)

history_files = []
for p in Path("results/tables").rglob("*.csv"):
    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    cols = set(df.columns)
    if "epoch" in cols and any(c in cols for c in ["loss", "train_loss", "val_cindex", "val_mean_cindex"]):
        history_files.append(p)

print(f"Found {len(history_files)} history-like CSV files.")

for p in history_files:
    df = pd.read_csv(p)
    rel = str(p).replace("/", "_").replace("\\", "_").replace(".csv", "")

    loss_cols = [c for c in ["loss", "train_loss", "val_loss", "supervised_loss", "pseudo_loss"] if c in df.columns]
    if "epoch" in df.columns and loss_cols:
        plt.figure(figsize=(7, 4))
        for c in loss_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(rel + " loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{rel}_loss.png", dpi=200)
        plt.close()

    val_cols = [c for c in df.columns if c.startswith("val") and ("cindex" in c or "c_index" in c)]
    if "epoch" in df.columns and val_cols:
        plt.figure(figsize=(7, 4))
        for c in val_cols:
            plt.plot(df["epoch"], df[c], label=c)
        plt.xlabel("Epoch")
        plt.ylabel("Validation metric")
        plt.title(rel + " validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{rel}_validation.png", dpi=200)
        plt.close()

print(f"Plots saved to: {out_dir}")
PY
}

echo "============================================================"
echo "Build required graphs"
echo "============================================================"

build_graph_if_missing \
    "Build synthetic_binary cosine graph" \
    "configs/gnn/deephit/synthetic_binary_k5_cosine.yaml" \
    "data/synthetic_binary/graphs/knn_k5_cosine_edge_index.npy" \
    "results/logs/build_synthetic_binary_k5_cosine.log"

build_graph_if_missing \
    "Build SEER k5 graph" \
    "configs/gnn/deephit/seer_k5.yaml" \
    "data/SEER/graphs/knn_k5_euclidean_edge_index.npy" \
    "results/logs/build_seer_k5.log"

echo "============================================================"
echo "Run new synthetic binary experiments"
echo "============================================================"

run_exp configs/gnn/deephit/synthetic_binary_k5_cosine.yaml
run_exp configs/gnn/deephit/synthetic_binary_k5_label20_cosine.yaml
run_exp configs/ssl/gnn_deephit/static_teacher/synthetic_binary_k5_label20_cosine.yaml
run_exp configs/gnn/deephit/synthetic_binary_k5_beta02.yaml

echo "============================================================"
echo "Run remaining synthetic competing-risk experiments"
echo "============================================================"

run_exp configs/gnn/deephit/synthetic_competing_k5_label20.yaml
run_exp configs/ssl/gnn_deephit/static_teacher/synthetic_competing_k5_label20.yaml

echo "============================================================"
echo "Run SEER and MNB baselines"
echo "============================================================"

run_exp configs/baselines/deepsurv/seer.yaml
run_exp configs/baselines/deephit/seer.yaml
run_exp configs/baselines/deepsurv/mnb_binary.yaml
run_exp configs/baselines/deephit/mnb_binary.yaml
run_exp configs/baselines/deephit/mnb_competing.yaml

echo "============================================================"
echo "Create plots"
echo "============================================================"

plot_histories 2>&1 | tee "results/logs/create_plots.log"
STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ]; then
    echo "FAILED: Create plots" | tee -a "$FAILED_FILE"
    echo "Log: results/logs/create_plots.log" >> "$FAILED_FILE"
    echo "" >> "$FAILED_FILE"
fi

echo "============================================================"
echo "Done"
echo "============================================================"

if [ -s "$FAILED_FILE" ]; then
    echo "Some steps failed. See:"
    echo "$FAILED_FILE"
    cat "$FAILED_FILE"
else
    echo "No failures recorded."
fi