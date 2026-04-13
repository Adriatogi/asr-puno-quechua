#!/usr/bin/env bash
# Run all three fine-tuning jobs sequentially inside Docker.
# Launch from project root:
#   docker compose run asr-puno-quechua bash finetune.sh
set -euo pipefail
mkdir -p logs

echo "=== ft_cpt_validated (Config 1: validated scripted + spont×3) ==="
bash training/scripts/run_finetune.sh \
    checkpoints/cpt/checkpoint_best.pt 4 checkpoints/ft_cpt_validated train \
    2>&1 | tee logs/finetune_ft_cpt_validated.log

echo "=== ft_cpt_silver (Config 2: validated scripted + silver spont) ==="
bash training/scripts/run_finetune.sh \
    checkpoints/cpt/checkpoint_best.pt 4 checkpoints/ft_cpt_silver train_silver \
    2>&1 | tee logs/finetune_ft_cpt_silver.log

echo "=== ft_xlsr_validated (XLSR baseline, Config 1) ==="
bash training/scripts/run_finetune.sh \
    checkpoints/xlsr2_300m.pt 4 checkpoints/ft_xlsr_validated train \
    2>&1 | tee logs/finetune_ft_xlsr_validated.log

echo "=== ft_xlsr_silver (XLSR baseline, Config 2: validated scripted + silver spont) ==="
bash training/scripts/run_finetune.sh \
    checkpoints/xlsr2_300m.pt 4 checkpoints/ft_xlsr_silver train_silver \
    2>&1 | tee logs/finetune_ft_xlsr_silver.log

echo "All four fine-tuning runs complete."
