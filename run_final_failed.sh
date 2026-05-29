#!/usr/bin/env bash

mkdir -p results/logs

FAILED_FILE="results/logs/final_failed_rerun_failures.txt"
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

echo "============================================================"
echo "1. MNB binary DeepHit"
echo "============================================================"

run_cmd \
    "Training MNB binary DeepHit" \
    "python scripts/train.py --config configs/baselines/deephit/mnb_binary.yaml" \
    "results/logs/final_mnb_binary_deephit_train.log"

run_cmd \
    "Evaluating MNB binary DeepHit" \
    "python scripts/evaluate.py --checkpoint results/checkpoints/baselines/deephit/mnb_binary.pt" \
    "results/logs/final_mnb_binary_deephit_eval.log"


echo "============================================================"
echo "2. Synthetic competing SSL"
echo "============================================================"

run_cmd \
    "Training synthetic competing SSL GraphSAGEDeepHit" \
    "python scripts/train.py --config configs/ssl/gnn_deephit/static_teacher/synthetic_competing_k5_label20.yaml" \
    "results/logs/final_synthetic_competing_ssl_train.log"

run_cmd \
    "Evaluating synthetic competing SSL GraphSAGEDeepHit" \
    "python scripts/evaluate.py --checkpoint results/checkpoints/ssl/gnn_deephit/static_teacher/synthetic_competing_k5_label20.pt" \
    "results/logs/final_synthetic_competing_ssl_eval.log"


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