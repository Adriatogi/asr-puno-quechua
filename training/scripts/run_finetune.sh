#!/usr/bin/env bash
# Fine-tuning launcher for Puno Quechua ASR.
# Run from project root. Must run convert_checkpoint.py on CPT output first.
#
# Usage:
#   bash training/scripts/run_finetune.sh <path/to/w2v.pt> [num_gpus] [save_dir]
#
# Examples:
#   bash training/scripts/run_finetune.sh checkpoints/cpt/checkpoint_best.pt 4
#   bash training/scripts/run_finetune.sh checkpoints/cpt/checkpoint_best.pt 4 checkpoints/ft_cpt
#   bash training/scripts/run_finetune.sh checkpoints/xlsr2_300m.pt 4 checkpoints/ft_xlsr_baseline

set -euo pipefail

ROOT=$(pwd)
W2V_PATH=$(realpath "${1:?"Usage: $0 <path/to/w2v.pt> [num_gpus] [save_dir]"}")
NUM_GPUS=${2:-1}
SAVE_DIR=$(realpath "${3:-"$ROOT/checkpoints/ft"}")

UPDATE_FREQ=$(( 2 / NUM_GPUS ))
# Clamp to at least 1
if [ "$UPDATE_FREQ" -lt 1 ]; then UPDATE_FREQ=1; fi

if [ ! -f "$W2V_PATH" ]; then
    echo "ERROR: Checkpoint not found: $W2V_PATH"
    echo "Run convert_checkpoint.py first:"
    echo "  python training/scripts/convert_checkpoint.py $W2V_PATH"
    exit 1
fi

echo "Starting fine-tuning:"
echo "  GPUs:          $NUM_GPUS"
echo "  W2V path:      $W2V_PATH"
echo "  Save dir:      $SAVE_DIR"
echo "  Update freq:   $UPDATE_FREQ"
echo ""

fairseq-hydra-train \
    --config-dir "$ROOT/training/configs" \
    --config-name w2v2-large-finetune_qxp \
    hydra.run.dir="/tmp/hydra/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
    task.data="$ROOT/data/manifests/finetune/qxp" \
    model.w2v_path="$W2V_PATH" \
    checkpoint.save_dir="$SAVE_DIR" \
    optimization.update_freq="[$UPDATE_FREQ]" \
    distributed_training.distributed_world_size=$NUM_GPUS
