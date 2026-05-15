#!/usr/bin/env bash
set -e

echo "Running synthetic experiments with overwrite enabled..."

CONFIGS=(
  # Baselines
  "configs/baselines/deepsurv/synthetic_binary.yaml"
  "configs/baselines/deephit/synthetic_binary.yaml"
  "configs/baselines/deephit/synthetic_binary_beta01.yaml"
  "configs/baselines/deephit/synthetic_competing.yaml"

  # GNN Cox
  "configs/gnn/cox/synthetic_binary.yaml"
  "configs/gnn/cox/synthetic_binary_k5.yaml"

  # GNN DeepHit
  "configs/gnn/deephit/synthetic_binary.yaml"
  "configs/gnn/deephit/synthetic_binary_k5.yaml"
  "configs/gnn/deephit/synthetic_binary_k5_beta01.yaml"
  "configs/gnn/deephit/synthetic_competing_k5.yaml"

  # Limited-label GNN DeepHit
  "configs/gnn/deephit/synthetic_binary_k5_label20.yaml"

  # SSL static teacher
  "configs/ssl/gnn_deephit/static_teacher/synthetic_binary_k5.yaml"
  "configs/ssl/gnn_deephit/static_teacher/synthetic_binary_k5_label20.yaml"
  "configs/ssl/gnn_deephit/static_teacher/synthetic_binary_k5_label20_beta01.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  if [ ! -f "$CONFIG" ]; then
    echo "Skipping missing config: $CONFIG"
    continue
  fi

  REL_PATH="${CONFIG#configs/}"
  REL_NO_EXT="${REL_PATH%.yaml}"

  CHECKPOINT="results/checkpoints/${REL_NO_EXT}.pt"

  echo ""
  echo "========================================"
  echo "Config: $CONFIG"
  echo "Checkpoint: $CHECKPOINT"
  echo "========================================"

  GRAPH_PATH=$(python - <<PY
import yaml
from pathlib import Path

config_path = Path("$CONFIG")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

graph = cfg.get("graph", {}) if cfg else {}
print(graph.get("edge_index_path", ""))
PY
)

  if [ -n "$GRAPH_PATH" ] && [ ! -f "$GRAPH_PATH" ]; then
    echo "Graph not found: $GRAPH_PATH"
    echo "Building graph..."
    python scripts/build_graphs.py --config "$CONFIG"
  elif [ -n "$GRAPH_PATH" ]; then
    echo "Graph exists: $GRAPH_PATH"
  fi

  echo "Training..."
  python scripts/train.py --config "$CONFIG"

  if [ -f "$CHECKPOINT" ]; then
    echo "Evaluating..."
    python scripts/evaluate.py --checkpoint "$CHECKPOINT"
  else
    echo "No checkpoint found after training. Skipping evaluation."
  fi
done

echo ""
echo "All synthetic experiments finished."