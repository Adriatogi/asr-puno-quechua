#!/usr/bin/env bash
# Run Omnilingual inference for both 300M model variants on test + test_spont,
# then compute WER/CER for each.
#
# Usage (from project root, conda activate asr-puno):
#   bash eval/omnilingual/run_all_omnilingual.sh
#
# Set FAIRSEQ2_CACHE_DIR before running if you need to redirect the model cache:
#   export FAIRSEQ2_CACHE_DIR=/matx/u/agamarra/.cache/fairseq2

set -euo pipefail

ROOT=$(pwd)

# CUDA compat symlink (libcudart.so.13 not installed, redirect to 12.8)
COMPAT_DIR="$HOME/cuda-compat"
mkdir -p "$COMPAT_DIR"
if [ ! -f "$COMPAT_DIR/libcudart.so.13" ]; then
    ln -sf /usr/local/cuda-12.8/lib64/libcudart.so.12 "$COMPAT_DIR/libcudart.so.13"
fi
export LD_LIBRARY_PATH="$COMPAT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Model cache — default to /matx to avoid AFS quota issues
export FAIRSEQ2_CACHE_DIR="${FAIRSEQ2_CACHE_DIR:-/matx/u/agamarra/.cache/fairseq2}"
mkdir -p "$FAIRSEQ2_CACHE_DIR"

MODELS=(omniASR_LLM_300M_v2 omniASR_CTC_300M_v2)
declare -A SUBSETS=(
    [test]="data/splits_joh/validated_scripted/test.tsv|data/wav/scripted"
    [test_spont]="data/splits_joh/validated_spontaneous/test_spontaneous.tsv|data/wav/spontaneous"
)

for model in "${MODELS[@]}"; do
    # Convert model card to a short directory name: omniASR_LLM_300M_v2 → llm_300m_v2
    dir_name=$(echo "$model" | sed 's/omniASR_//;s/_v2$/_v2/' | tr '[:upper:]' '[:lower:]')
    out_dir="$ROOT/results/omnilingual/$dir_name"
    mkdir -p "$out_dir"

    for subset in test test_spont; do
        IFS="|" read -r tsv audio_dir <<< "${SUBSETS[$subset]}"
        output="$out_dir/${subset}_transcribed.tsv"

        if [ -f "$output" ]; then
            echo "=== SKIP (already exists): $model / $subset ==="
            continue
        fi

        echo "=== $model / $subset ==="
        python "$ROOT/eval/omnilingual/run_omnilingual.py" \
            --tsv "$ROOT/$tsv" \
            --audio_dir "$ROOT/$audio_dir" \
            --output "$output" \
            --model_card "$model"
        echo ""
    done
done

echo "=== Computing WER/CER ==="
for model in "${MODELS[@]}"; do
    dir_name=$(echo "$model" | sed 's/omniASR_//;s/_v2$/_v2/' | tr '[:upper:]' '[:lower:]')
    out_dir="$ROOT/results/omnilingual/$dir_name"

    for subset in test test_spont; do
        output="$out_dir/${subset}_transcribed.tsv"
        if [ -f "$output" ]; then
            echo "--- $model / $subset ---"
            python "$ROOT/eval/omnilingual/compute_wer.py" "$output"
        fi
    done
done

echo ""
echo "Done. Results in results/omnilingual/"
