# Puno Quechua ASR — Inference Setup

Transcribe Puno Quechua audio using `ft_cpt_validated` (1.22% WER on scripted speech, 13.59% on spontaneous/conversational).

---

## Requirements

- Docker (with GPU support)
- ~4GB disk for the model checkpoint

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd asr-puno-quechua
```

**2. Download the checkpoint**

```bash
pip install huggingface_hub
huggingface-cli download Adriatogi/wav2vec2-puno-quechua-ft-cpt-validated \
    checkpoint_best.pt \
    --local-dir checkpoints/ft_cpt_validated/
```

**3. Configure your GPU**

Copy `.env.example` to `.env` and set your GPU indices:

```bash
cp .env.example .env
# Edit .env: set DOCKER_CUDA_VISIBLE_DEVICES to your GPU index, e.g. DOCKER_CUDA_VISIBLE_DEVICES=0
```

---

## Usage

All commands are run from the project root. Docker handles all dependencies.

**Transcribe one or more files**

```bash
docker compose run asr-puno-quechua -c "python colleague_inference/transcribe.py recording.wav"
docker compose run asr-puno-quechua -c "python colleague_inference/transcribe.py file1.wav file2.wav file3.wav"
```

Output:
```
recording.wav  →  iskay urququnaq chaupinpi payqa tiyan
```

**Transcribe a whole folder (writes TSV)**

```bash
docker compose run asr-puno-quechua -c \
  "python colleague_inference/transcribe.py --input_dir ./my_audio/ --output_tsv results.tsv"
```

Output TSV columns: `path`, `transcription`

**Transcribe from a TSV manifest**

If you have a TSV with a `path` column:

```bash
docker compose run asr-puno-quechua -c \
  "python colleague_inference/transcribe.py --tsv my_manifest.tsv --output_tsv results.tsv"
```

---

## Notes

- **Audio format**: WAV or MP3, any sample rate, mono or stereo — converted to 16kHz mono automatically
- **GPU**: used automatically if available; falls back to CPU (slower but works)
- **Accuracy**: 1.22% WER on scripted Puno Quechua; 13.59% on spontaneous speech
- **Checkpoint**: `--ckpt PATH` overrides the default checkpoint location if needed
