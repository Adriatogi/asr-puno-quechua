#!/usr/bin/env bash
# Continued Pre-Training (CPT) launcher for Puno Quechua.
# Run from project root.
#
# Usage:
#   bash training/scripts/run_cpt.sh [num_gpus] [train_subset]
#
# Examples:
#   bash training/scripts/run_cpt.sh 1 "qxp_scripted,qxp_spontaneous"
#   bash training/scripts/run_cpt.sh 4 "qxp_scripted,qxp_spontaneous"
#   bash training/scripts/run_cpt.sh 1 "qxp_scripted"

set -euo pipefail

ROOT=$(pwd)
NUM_GPUS=${1:-1}
TRAIN_SUBSET=${2:-"qxp_scripted,qxp_spontaneous"}

# Gradient accumulation scales inversely with GPU count to keep effective batch size constant.
# Base: update_freq=16 on 1 GPU → effective batch = 16 × max_tokens
UPDATE_FREQ=$(( 16 / NUM_GPUS ))

CHECKPOINT_PATH="$ROOT/checkpoints/xlsr2_300m.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: XLSR-128 checkpoint not found at $CHECKPOINT_PATH"
    echo "Download it with:"
    echo "  wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -P checkpoints/"
    exit 1
fi

echo "Starting CPT:"
echo "  GPUs:          $NUM_GPUS"
echo "  Train subset:  $TRAIN_SUBSET"
echo "  Update freq:   $UPDATE_FREQ"
echo "  Checkpoint:    $CHECKPOINT_PATH"
echo ""

fairseq-hydra-train \
    --config-dir "$ROOT/training/configs" \
    --config-name w2v2-large-cpt_qxp \
    common.user_dir="$ROOT/training/custom_task" \
    task.data="$ROOT/data/manifests/pretrain/" \
    dataset.train_subset="$TRAIN_SUBSET" \
    optimization.update_freq="[$UPDATE_FREQ]" \
    distributed_training.distributed_world_size=$NUM_GPUS \
    checkpoint.finetune_from_model="$CHECKPOINT_PATH"
