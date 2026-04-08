#!/usr/bin/env bash
# Evaluate a fine-tuned Puno Quechua ASR model.
# Run from project root.
#
# Usage:
#   bash training/scripts/evaluate.sh <path/to/finetuned.pt> [results_dir] [manifest_dir]
#
# Examples:
#   bash training/scripts/evaluate.sh checkpoints/ft/checkpoint_best.pt results/
#   bash training/scripts/evaluate.sh checkpoints/ft/checkpoint_best.pt results/additional data/manifests/additional

set -euo pipefail

ROOT=$(pwd)
FT_CHECKPOINT=${1:?"Usage: $0 <path/to/finetuned.pt> [results_dir] [manifest_dir]"}
RESULTS_DIR=${2:-"$ROOT/results"}
MANIFEST_DIR=${3:-"$ROOT/data/manifests/finetune/qxp"}

if [ ! -f "$FT_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $FT_CHECKPOINT"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Evaluating: $FT_CHECKPOINT"
echo "Results:    $RESULTS_DIR"
echo ""

python -m fairseq_cli.speech_recognition.infer \
    "$MANIFEST_DIR" \
    --gen-subset test \
    --path "$FT_CHECKPOINT" \
    --results-path "$RESULTS_DIR" \
    --task audio_finetuning \
    --nbest 1 \
    --w2l-decoder viterbi \
    --criterion ctc \
    --labels ltr \
    --max-tokens 5000000 \
    --post-process letter

echo ""
echo "WER results saved to $RESULTS_DIR"
grep "WER" "$RESULTS_DIR"/hypo.units* 2>/dev/null || true
