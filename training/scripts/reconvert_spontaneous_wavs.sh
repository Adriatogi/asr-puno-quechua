#!/usr/bin/env bash
# Re-convert all spontaneous MP3 → WAV using ffmpeg on the host.
# Fixes libsndfile "System error" in Docker caused by non-standard WAV headers
# written by older torchaudio/soundfile versions.
# Run from project root (no Docker needed):
#   bash training/scripts/reconvert_spontaneous_wavs.sh [num_workers]
set -euo pipefail

MP3_DIR="data/spontaneous/sps-corpus-3.0-2026-03-09-qxp/audios"
WAV_DIR="data/wav/spontaneous"
WORKERS=${1:-8}

if [ ! -d "$MP3_DIR" ]; then
    echo "ERROR: MP3 source dir not found: $MP3_DIR"
    exit 1
fi

mkdir -p "$WAV_DIR"
mp3_count=$(ls "$MP3_DIR"/*.mp3 2>/dev/null | wc -l)
echo "Re-converting $mp3_count MP3s → 16kHz mono WAV (${WORKERS} workers) ..."

ls "$MP3_DIR"/*.mp3 | xargs -P "$WORKERS" -I{} bash -c '
    mp3="$1"
    base=$(basename "$mp3" .mp3)
    wav="'"$WAV_DIR"'/${base}.wav"
    ffmpeg -y -i "$mp3" -ar 16000 -ac 1 -sample_fmt s16 "$wav" -loglevel error
' _ {}

echo "Done. $mp3_count WAVs written to $WAV_DIR"
echo ""
echo "Refresh nframes in manifests before training:"
echo "  python training/scripts/build_finetune_manifests.py"
