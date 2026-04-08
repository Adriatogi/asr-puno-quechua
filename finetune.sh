#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

#docker compose run asr-puno-quechua
bash training/scripts/run_finetune.sh checkpoints/cpt/checkpoint_best.pt 4 checkpoints/ft_cpt 2>&1 | tee logs/finetune_ft_cpt.log

bash training/scripts/run_finetune.sh checkpoints/xlsr2_300m.pt 4 checkpoints/ft_xlsr 2>&1 | tee logs/finetune_ft_xlsr.log
