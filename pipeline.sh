#!/usr/bin/env bash
# =============================================================================
# Puno Quechua ASR — Full Pipeline
# =============================================================================
# Documents every step needed to reproduce results from scratch.
# NOT meant to be run top-to-bottom in one shot — some steps run on the host,
# some inside Docker, and some take hours. Read each section before running.
#
# Conventions:
#   [HOST]   — run on the host machine (conda activate asr-puno)
#   [DOCKER] — run inside Docker (docker compose run asr-puno-quechua bash -c "...")
#
# Prerequisites:
#   - data/splits_joh/ TSVs from colleague (pre-split Common Voice + spontaneous)
#   - Raw MP3s in data/scripted/ and data/spontaneous/
#   - CPT checkpoint at checkpoints/cpt/checkpoint_best.pt
#   - XLSR checkpoint at checkpoints/xlsr2_300m.pt
#   - Docker image built and docker compose configured
#   - conda env asr-puno with: pandas, soundfile, jiwer, fairseq2, omnilingual-asr
# =============================================================================

set -euo pipefail
ROOT=$(pwd)

# =============================================================================
# STEP 1 — Audio conversion (HOST)
# Convert scripted MP3 → 16kHz mono WAV
# =============================================================================
# [HOST] Already done. If re-running:
#
# Scripted (Common Voice MP3s):
#   mkdir -p data/wav/scripted
#   for f in data/scripted/cv-corpus-*/clips/*.mp3; do
#       base=$(basename "$f" .mp3)
#       ffmpeg -y -i "$f" -ar 16000 -ac 1 -sample_fmt s16 \
#           data/wav/scripted/${base}.wav -loglevel error
#   done
#
# Spontaneous:
#   bash training/scripts/reconvert_spontaneous_wavs.sh
#
# Outputs: data/wav/scripted/*.wav, data/wav/spontaneous/*.wav

echo "[STEP 1] Audio conversion — assumed already done."
echo "  Scripted WAVs: data/wav/scripted/"
echo "  Spontaneous WAVs: data/wav/spontaneous/"
echo ""

# =============================================================================
# STEP 2 — Build fine-tuning manifests (HOST)
# Reads data/splits_joh/ TSVs, normalizes text, filters >30s clips,
# writes fairseq .tsv + .ltr manifests to data/manifests/finetune/qxp_v2/
# =============================================================================
# [HOST]
#   python training/scripts/build_finetune_manifests.py
#
# Outputs:
#   data/manifests/finetune/qxp_v2/train.tsv + train.ltr      (18,872 clips)
#   data/manifests/finetune/qxp_v2/train_silver.tsv + .ltr    (22,387 clips)
#   data/manifests/finetune/qxp_v2/valid.tsv + .ltr
#   data/manifests/finetune/qxp_v2/test.tsv + .ltr            (2,296 clips)
#   data/manifests/finetune/qxp_v2/test_spont.tsv + .ltr      (360 clips)
#   data/manifests/finetune/qxp_v2/dict.ltr.txt               (45 chars)
#
# NOTE: dict.ltr.txt includes '|' (word boundary token). If re-running on
# existing checkpoints trained without '|' in the dict, see code review in
# the plan — analyze_results.py handles the <unk>-as-boundary case correctly.

echo "[STEP 2] Build manifests — assumed already done."
echo "  Manifests: data/manifests/finetune/qxp_v2/"
echo ""

# =============================================================================
# STEP 3 — Fine-tuning (DOCKER)
# Four models: CPT and XLSR × validated and silver training data
# =============================================================================
# [DOCKER]
#   docker compose run asr-puno-quechua bash finetune.sh
#
# Or individually (e.g., to add ft_xlsr_silver later):
#   docker compose run asr-puno-quechua bash -c "
#     bash training/scripts/run_finetune.sh \
#       checkpoints/xlsr2_300m.pt 4 checkpoints/ft_xlsr_silver train_silver \
#       2>&1 | tee logs/finetune_ft_xlsr_silver.log
#   "
#
# Config: training/configs/w2v2-large-finetune_qxp.yaml
#   - 20,000 updates, LR=5e-5, tri-stage schedule
#   - Encoder frozen for first 10,000 updates
#   - CTC loss, best checkpoint by dev WER
#
# Outputs:
#   checkpoints/ft_cpt_validated/checkpoint_best.pt
#   checkpoints/ft_cpt_silver/checkpoint_best.pt
#   checkpoints/ft_xlsr_validated/checkpoint_best.pt
#   checkpoints/ft_xlsr_silver/checkpoint_best.pt

echo "[STEP 3] Fine-tuning — run inside Docker:"
echo "  docker compose run asr-puno-quechua bash finetune.sh"
echo ""

# =============================================================================
# STEP 4 — Fairseq inference (DOCKER)
# Runs all 4 models × 2 test subsets, writes hypo/ref files
# =============================================================================
# [DOCKER]
#   docker compose run asr-puno-quechua bash eval/fairseq/run_eval.sh
#
# Uses eval/fairseq/infer_patched.py which patches two missing flashlight symbols:
#   - CriterionType (stub Enum)
#   - W2lViterbiDecoder.decode (replaced with PyTorch argmax greedy CTC)
#
# Outputs per model/subset, e.g. results/ft_cpt_validated/test/:
#   hypo.word-checkpoint_best.pt-test.txt   (model output, words sep by <unk>)
#   ref.word-checkpoint_best.pt-test.txt    (reference, same format)
#   hypo.units-* / ref.units-*             (character-level, for CER)
#
# WARNING: The .wer files fairseq writes report Sentence Error Rate (SER),
# NOT word WER. Do not cite those numbers. Use Step 5 instead.

echo "[STEP 4] Fairseq inference — run inside Docker:"
echo "  docker compose run asr-puno-quechua bash eval/fairseq/run_eval.sh"
echo ""

# =============================================================================
# STEP 5 — WER computation for fairseq models (HOST)
# Parses hypo/ref files, computes true micro word-level WER + CER
# =============================================================================
# [HOST]
#   conda activate asr-puno
#   python eval/fairseq/analyze_results.py --results-dir results
#
# Outputs: results/summary.csv
#
# How it works:
#   - Parses "word" files: <text> (None-<id>) format
#   - Replaces <unk> → space to recover word boundaries (since | not in dict)
#   - jiwer.wer(ref_list, hypo_list) = micro-WER (total edits / total words)
#   - jiwer.cer() from units files for CER

echo "[STEP 5] WER computation:"
echo "  python eval/fairseq/analyze_results.py --results-dir results"
echo ""

# =============================================================================
# STEP 6 — Omnilingual baseline (HOST)
# Zero-shot inference with omniASR_LLM_300M_v2 and omniASR_CTC_300M_v2
# =============================================================================
# [HOST]
#   conda activate asr-puno
#   export FAIRSEQ2_CACHE_DIR=/matx/u/agamarra/.cache/fairseq2
#   bash eval/omnilingual/run_all_omnilingual.sh
#
# Requires CUDA compat symlink (handled automatically by the script):
#   ~/cuda-compat/libcudart.so.13 → /usr/local/cuda-12.8/lib64/libcudart.so.12
#
# Outputs:
#   results/omnilingual/llm_300m_v2/test_transcribed.tsv
#   results/omnilingual/llm_300m_v2/test_spont_transcribed.tsv
#   results/omnilingual/ctc_300m_v2/test_transcribed.tsv
#   results/omnilingual/ctc_300m_v2/test_spont_transcribed.tsv
#   (+ _scored.tsv variants with per-utterance gold column)
#
# NOTE: both gold and transcription are normalized (lowercase + strip ?!¿¡.,)
# before WER computation in compute_wer.py.

echo "[STEP 6] Omnilingual baseline:"
echo "  export FAIRSEQ2_CACHE_DIR=/matx/u/agamarra/.cache/fairseq2"
echo "  bash eval/omnilingual/run_all_omnilingual.sh"
echo ""

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo "================================================================="
echo "Results (true micro word-level WER):"
echo ""
echo "  Source: results/summary.csv (fairseq models)"
echo "          results/omnilingual/*/  (omnilingual baselines)"
echo ""
echo "  Model                  test WER   test_spont WER"
echo "  ft_cpt_validated        1.22%      13.59%"
echo "  ft_cpt_silver           2.29%       3.37%"
echo "  ft_xlsr_validated       2.40%      13.25%"
echo "  ft_xlsr_silver          —           —        (pending)"
echo "  omniASR_LLM_300M_v2    7.07%      11.98%"
echo "  omniASR_CTC_300M_v2   24.47%      20.22%"
echo "================================================================="
