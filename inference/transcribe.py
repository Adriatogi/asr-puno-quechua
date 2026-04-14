#!/usr/bin/env python3
"""
Transcribe Puno Quechua audio with ft_cpt_validated (1.22% WER scripted).
Run inside Docker from the project root.

Single files:
  docker compose run asr-puno-quechua -c "python inference/transcribe.py recording.wav"
  docker compose run asr-puno-quechua -c "python inference/transcribe.py *.wav"

Batch (writes TSV):
  docker compose run asr-puno-quechua -c "python inference/transcribe.py --input_dir ./my_audio/ --output_tsv results.tsv"
  docker compose run asr-puno-quechua -c "python inference/transcribe.py --tsv manifest.tsv --output_tsv results.tsv"
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/fairseq")
warnings.filterwarnings("ignore")

import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT     = Path(__file__).resolve().parent.parent
CKPT     = ROOT / "checkpoints" / "ft_cpt_validated" / "checkpoint_best.pt"
DICT_DIR = ROOT / "data" / "manifests" / "finetune" / "qxp_v2"


def load_model(ckpt_path, device):
    from fairseq import checkpoint_utils
    print(f"Loading model...")
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(ckpt_path)],
        arg_overrides={"data": str(DICT_DIR)},
    )
    model = models[0].eval().to(device)
    print("Ready.\n")
    return model, task


def load_audio(path):
    """Load any audio file, convert to 16kHz mono float32."""
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)          # stereo → mono
    if sr != 16000:
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)


def transcribe_one(model, task, audio, device):
    source = torch.FloatTensor(audio).unsqueeze(0).to(device)
    padding_mask = torch.zeros(1, source.shape[-1], dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(source=source, padding_mask=padding_mask)

    logits = out["encoder_out"]       # (T, B, C)
    ids = logits.argmax(dim=-1)[:, 0].tolist()  # (T,)

    # CTC collapse: remove consecutive duplicates, then remove blank (bos = index 0)
    blank = task.target_dictionary.bos()
    prev, tokens = None, []
    for id in ids:
        if id != prev:
            if id != blank:
                tokens.append(id)
        prev = id

    # Indices → characters. <unk> (index 3) = word boundary for this model.
    chars = "".join(task.target_dictionary[t] for t in tokens)
    return chars.replace("<unk>", " ").strip()


def collect_paths(args):
    if args.files:
        return [Path(f) for f in args.files]
    if args.input_dir:
        d = Path(args.input_dir)
        return sorted(d.glob("*.wav")) + sorted(d.glob("*.mp3"))
    if args.tsv:
        with open(args.tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            return [Path(row["path"]) for row in reader]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Audio file(s) to transcribe")
    parser.add_argument("--input_dir", help="Directory of .wav/.mp3 files")
    parser.add_argument("--tsv", help="TSV with a 'path' column")
    parser.add_argument("--output_tsv", help="Write results to this TSV file")
    parser.add_argument("--ckpt", default=str(CKPT), help="Path to checkpoint_best.pt")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, task = load_model(args.ckpt, device)

    paths = collect_paths(args)
    if not paths:
        parser.error("Provide audio files, --input_dir, or --tsv")

    results = []
    for path in paths:
        try:
            audio = load_audio(path)
            text  = transcribe_one(model, task, audio, device)
            results.append((str(path), text))
            if not args.output_tsv:
                print(f"{Path(path).name}  →  {text}")
        except Exception as e:
            print(f"WARNING: skipping {path}: {e}", file=sys.stderr)
            results.append((str(path), "ERROR"))

    if args.output_tsv:
        out = Path(args.output_tsv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["path", "transcription"])
            writer.writerows(results)
        print(f"\nSaved {len(results)} transcriptions → {out}")


if __name__ == "__main__":
    main()
