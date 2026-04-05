# asr-puno-quechua

Automatic speech recognition for Puno Quechua (qxp) using Mozilla Common Voice data.

## Datasets

Both datasets are from the [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) and cover Puno Quechua (ISO 639-3: qxp). Licensed CC0-1.0.

| | Scripted Speech 25.0 | Spontaneous Speech 3.0 |
|---|---|---|
| **Speech type** | Speakers read pre-written sentences | Speakers respond naturally to prompts |
| **Clips** | 25,382 | 7,286 |
| **Validated** | 22,727 (89.5%) | 1,074 (14.7%) |
| **Validated duration** | 31.2 hours | 5.2 hours |
| **Speakers** | 81 | 110 |
| **Best for** | ASR model training | Natural prosody, language patterns |

**Scripted speech** is clearer and easier to transcribe — speakers read known sentences, so transcripts are exact. **Spontaneous speech** captures how people actually talk: natural hesitations, varied sentence structure, informal vocabulary. It's harder to transcribe but more representative of real-world use.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

Get an API key from your profile at [datacollective.mozillafoundation.org](https://datacollective.mozillafoundation.org/) (Profile > API), then:

```bash
cp .env.example .env
# fill in your key in .env
python data/download_data.py
```

Data is saved to `data/scripted/` and `data/spontaneous/`.

## Explore

```bash
jupyter notebook data/explore.ipynb
```

Covers demographics, transcript lengths, clip durations, and interactive waveform/spectrogram/audio playback for both datasets.
