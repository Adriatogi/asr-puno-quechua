# Training — Puno Quechua ASR

wav2vec 2.0 continued pre-training (CPT) → CTC fine-tuning → evaluation pipeline for Puno Quechua (qxp).

## Why this pipeline

| Stage | What it does | Data used |
|---|---|---|
| **CPT** | Adapts XLSR-128 to Quechua acoustics via self-supervised contrastive loss | All ~66h audio (no transcriptions needed) |
| **Fine-tune** | Trains a CTC head to map acoustic representations → letters | ~32h validated + transcribed clips |
| **Eval** | Computes WER on held-out test set | Test split manifests |

Starting from XLSR-128 (Meta's model pre-trained on 128 languages) and doing CPT first means the model learns Quechua phonology before ever seeing a transcript — crucial for a low-resource language.

## Checkpoint flow

```
checkpoints/xlsr2_300m.pt          ← download once (Meta XLSR-128, 3.8GB)
        ↓  run_cpt.sh
checkpoints/cpt/checkpoint_best.pt ← CPT output (task name = temp_sampled_audio_pretraining)
        ↓  convert_checkpoint.py
checkpoints/cpt/checkpoint_best.pt ← fixed in-place (task name = audio_pretraining)
        ↓  run_finetune.sh
checkpoints/ft/checkpoint_best.pt  ← ready for inference
        ↓  evaluate.sh
results/                            ← WER output
```

## Setup

Training requires Docker with GPU support. The `fauxneticien/fairseq-asr` image includes fairseq and all dependencies.

```bash
# Start Docker container (from project root)
docker-compose run asr-puno-quechua

# Inside the container, download XLSR-128 base checkpoint (~3.8 GB)
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -P checkpoints/
```

All subsequent training commands should be run inside the Docker container.

**Multi-GPU support:** The Docker container has access to all GPUs on your host. To use multiple GPUs, simply change the `NUM_GPUS` parameter in the training commands (see examples in Steps 2 and 4 below). The scripts automatically adjust gradient accumulation to maintain constant effective batch size.

## Step 1 — Prepare data

Converts MP3 → WAV (16kHz mono), builds pre-train manifests and fine-tune manifests + letter files.

```bash
# Both datasets (recommended)
python training/scripts/prepare_cv_qxp_data.py

# Scripted only
python training/scripts/prepare_cv_qxp_data.py --splits scripted

# Skip conversion if WAVs already exist
python training/scripts/prepare_cv_qxp_data.py --skip-conversion
```

**Outputs:**
```
data/wav/scripted/              ← converted WAVs
data/wav/spontaneous/
data/manifests/pretrain/
    qxp_scripted.tsv
    qxp_spontaneous.tsv
    qxp_valid.tsv
data/manifests/finetune/qxp/
    train.tsv / valid.tsv / test.tsv
    train.ltr / valid.ltr / test.ltr
    train_spont.tsv / ...       ← spontaneous validated clips (if any per split)
    dict.ltr.txt
```

## Step 2 — Continued Pre-Training (CPT)

```bash
# 1 GPU
bash training/scripts/run_cpt.sh 1 "qxp_scripted,qxp_spontaneous"

# 4 GPUs
bash training/scripts/run_cpt.sh 4 "qxp_scripted,qxp_spontaneous"

# Scripted only
bash training/scripts/run_cpt.sh 1 "qxp_scripted"
```

Checkpoints saved to `checkpoints/cpt/`. Training runs for 10,000 updates (~few hours on L40S).

## Step 3 — Convert checkpoint

Must be done before fine-tuning:

```bash
python training/scripts/convert_checkpoint.py checkpoints/cpt/checkpoint_best.pt
```

This changes the task name inside the checkpoint from `temp_sampled_audio_pretraining` → `audio_pretraining` so fairseq's fine-tuning task can load it.

## Step 4 — Fine-tuning

```bash
# 1 GPU
bash training/scripts/run_finetune.sh checkpoints/cpt/checkpoint_best.pt 1

# 4 GPUs
bash training/scripts/run_finetune.sh checkpoints/cpt/checkpoint_best.pt 4
```

Checkpoints saved to `checkpoints/ft/`. Best checkpoint selected by WER on validation set.

## Step 5 — Evaluate

```bash
bash training/scripts/evaluate.sh checkpoints/ft/checkpoint_best.pt
```

WER results saved to `results/`.

## GPU scaling reference

| GPUs | update_freq (CPT) | update_freq (FT) | Notes |
|---|---|---|---|
| 1 | 16 | 2 | Default |
| 2 | 8 | 1 | |
| 4 | 4 | 1 | |
| 8 | 2 | 1 | |

`update_freq` is set automatically by the run scripts. Effective batch size stays constant.

## Directory structure

```
training/
├── configs/
│   ├── w2v2-large-cpt_qxp.yaml       # CPT config (XLSR-128 → qxp)
│   └── w2v2-large-finetune_qxp.yaml  # Fine-tune config (CTC)
├── custom_task/                        # Custom fairseq task for multilingual CPT
│   └── tasks/temp_sampled_audio_pretraining.py
└── scripts/
    ├── prepare_cv_qxp_data.py          # Data prep (MP3→WAV, manifests, .ltr)
    ├── convert_checkpoint.py           # Fix task name after CPT
    ├── run_cpt.sh                      # CPT launcher
    ├── run_finetune.sh                 # Fine-tune launcher
    └── evaluate.sh                     # WER evaluation
```
