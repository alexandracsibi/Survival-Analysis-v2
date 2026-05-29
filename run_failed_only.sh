#!/usr/bin/env bash

mkdir -p results/logs
mkdir -p results/figures/auto

FAILED_FILE="results/logs/failed_rerun_only.txt"
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

run_train_eval () {
    CONFIG="$1"
    CHECKPOINT="$2"

    if [ ! -f "$CONFIG" ]; then
        echo "SKIP: config does not exist: $CONFIG" | tee -a "$FAILED_FILE"
        return 0
    fi

    NAME=$(echo "$CONFIG" | sed 's#configs/##' | sed 's#/#_#g' | sed 's#.yaml##')

    run_cmd \
        "Training $CONFIG" \
        "python scripts/train.py --config $CONFIG" \
        "results/logs/${NAME}_rerun_train.log"

    if [ -f "$CHECKPOINT" ]; then
        run_cmd \
            "Evaluating $CHECKPOINT" \
            "python scripts/evaluate.py --checkpoint $CHECKPOINT" \
            "results/logs/${NAME}_rerun_eval.log"
    else
        echo "SKIP evaluation: checkpoint not found: $CHECKPOINT" | tee -a "$FAILED_FILE"
    fi
}

run_eval_only () {
    CHECKPOINT="$1"
    NAME="$2"

    if [ -f "$CHECKPOINT" ]; then
        run_cmd \
            "Evaluating $CHECKPOINT" \
            "python scripts/evaluate.py --checkpoint $CHECKPOINT" \
            "results/logs/${NAME}_rerun_eval.log"
    else
        echo "SKIP evaluation: checkpoint not found: $CHECKPOINT" | tee -a "$FAILED_FILE"
    fi
}

echo "============================================================"
echo "1. Rerun failed SEER evaluations"
echo "============================================================"

run_eval_only \
    "results/checkpoints/baselines/deepsurv/seer.pt" \
    "baselines_deepsurv_seer"

run_eval_only \
    "results/checkpoints/baselines/deephit/seer.pt" \
    "baselines_deephit_seer"


echo "============================================================"
echo "2. Rerun failed MNB binary DeepHit"
echo "============================================================"

run_train_eval \
    "configs/baselines/deephit/mnb_binary.yaml" \
    "results/checkpoints/baselines/deephit/mnb_binary.pt"


echo "============================================================"
echo "3. Rerun missing/failed synthetic competing SSL"
echo "============================================================"

run_train_eval \
    "configs/ssl/gnn_deephit/static_teacher/synthetic_competing_k5_label20.yaml" \
    "results/checkpoints/ssl/gnn_deephit/static_teacher/synthetic_competing_k5_label20.pt"


echo "============================================================"
echo "4. Build SEER graph last"
echo "============================================================"

if [ -f "data/SEER/graphs/knn_k5_euclidean_edge_index.npy" ]; then
    echo "SKIP: SEER graph already exists."
else
    run_cmd \
        "Build SEER k5 graph" \
        "python scripts/build_graphs.py --config configs/gnn/deephit/seer_k5.yaml" \
        "results/logs/build_seer_k5_rerun.log"
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