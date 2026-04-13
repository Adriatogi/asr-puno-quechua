"""
Prepare data/additional_data/ for evaluation against trained models.

Creates:
  data/additional_data.tsv          <- for run_omnilingual.py (path + sentence)
  data/additional_data_16k/*.wav    <- resampled to 16kHz (from 44.1kHz originals)
  data/manifests/additional/
      test.tsv                      <- fairseq manifest for evaluate.sh
      dict.ltr.txt                  <- copied from finetune/qxp/

Usage (from project root):
  python eval/omnilingual/prepare_additional_data.py
"""

import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
import torchaudio

ROOT           = Path(__file__).resolve().parents[1]
ADDITIONAL_DIR = ROOT / "data" / "additional_data"
RESAMPLED_DIR  = ROOT / "data" / "additional_data_16k"
OUT_TSV        = ROOT / "data" / "additional_data.tsv"
MANIFEST_DIR   = ROOT / "data" / "manifests" / "additional"
DICT_SRC       = ROOT / "data" / "manifests" / "finetune" / "qxp" / "dict.ltr.txt"
TARGET_SR      = 16000


def main():
    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(ADDITIONAL_DIR.glob("*.wav"))
    pairs, missing_txt = [], []
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt))
        else:
            missing_txt.append(wav.name)

    if missing_txt:
        print(f"Warning: {len(missing_txt)} WAV files with no matching .txt: "
              f"{missing_txt[:3]}{'...' if len(missing_txt) > 3 else ''}")

    print(f"Found {len(pairs)} paired WAV + TXT files")

    rows, manifest_entries = [], []
    for wav_path, txt_path in pairs:
        sentence = txt_path.read_text(encoding="utf-8").strip()

        out_wav = RESAMPLED_DIR / wav_path.name
        if not out_wav.exists():
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
            torchaudio.save(str(out_wav), waveform, TARGET_SR)

        num_frames = sf.info(str(out_wav)).frames
        rows.append({"path": wav_path.name, "sentence": sentence})
        manifest_entries.append((wav_path.name, num_frames))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Wrote {len(df)} rows -> {OUT_TSV.relative_to(ROOT)}")

    manifest_tsv = MANIFEST_DIR / "test.tsv"
    with open(manifest_tsv, "w") as f:
        f.write(str(RESAMPLED_DIR) + "\n")
        for fname, nframes in manifest_entries:
            f.write(f"{fname}\t{nframes}\n")
    print(f"Wrote fairseq manifest -> {manifest_tsv.relative_to(ROOT)}")

    if DICT_SRC.exists():
        shutil.copy(DICT_SRC, MANIFEST_DIR / "dict.ltr.txt")
        print(f"Copied dict.ltr.txt -> {(MANIFEST_DIR / 'dict.ltr.txt').relative_to(ROOT)}")
    else:
        print(f"Warning: dict.ltr.txt not found at {DICT_SRC} — run prepare_cv_qxp_data.py first")


if __name__ == "__main__":
    main()
