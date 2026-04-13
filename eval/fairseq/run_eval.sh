#!/usr/bin/env bash
# Run fairseq evaluation for all fine-tuned models × test subsets.
# Launch from project root inside Docker:
#   docker compose run asr-puno-quechua bash eval/fairseq/run_eval.sh
set -euo pipefail

ROOT=$(pwd)
MANIFEST_DIR="$ROOT/data/manifests/finetune/qxp_v2"

declare -A MODELS=(
    [ft_cpt_validated]="checkpoints/ft_cpt_validated/checkpoint_best.pt"
    [ft_cpt_silver]="checkpoints/ft_cpt_silver/checkpoint_best.pt"
    [ft_xlsr_validated]="checkpoints/ft_xlsr_validated/checkpoint_best.pt"
    [ft_xlsr_silver]="checkpoints/ft_xlsr_silver/checkpoint_best.pt"
)

SUBSETS=(test test_spont)

for name in "${!MODELS[@]}"; do
    ckpt="${MODELS[$name]}"
    if [ ! -f "$ckpt" ]; then
        echo "WARNING: checkpoint not found, skipping: $ckpt"
        continue
    fi
    for subset in "${SUBSETS[@]}"; do
        echo "=== $name / $subset ==="
        bash eval/fairseq/evaluate.sh \
            "$ckpt" \
            "results/$name/$subset" \
            "$MANIFEST_DIR" \
            "$subset" \
            2>&1 | tee "logs/eval_${name}_${subset}.log"
        echo ""
    done
done

echo "All evaluations complete. Run:"
echo "  python eval/fairseq/analyze_results.py"
