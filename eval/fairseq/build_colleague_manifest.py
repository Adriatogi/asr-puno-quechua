"""
Build a fairseq manifest from the colleague's eval TSV
(omni_transcripts_for_eval_small.tsv) so ft_cpt_silver and
ft_xlsr_silver can be evaluated on it.

Usage (from project root):
  python eval/fairseq/build_colleague_manifest.py

Output: data/manifests/colleague_eval_small/
  colleague_eval_small.tsv + .ltr
"""

from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
TSV_IN  = ROOT / "data" / "splits_joh" / "omni_transcripts_for_eval_small.tsv"
WAV_DIR = ROOT / "data" / "wav" / "scripted"
OUT_DIR = ROOT / "data" / "manifests" / "colleague_eval_small"
DOCKER_WAV_ROOT = Path("/workspace/data/wav")


def to_ltr(text: str) -> str:
    return " ".join("|" if c == " " else c for c in text)


df = pd.read_csv(TSV_IN, sep="\t", dtype=str, keep_default_na=False)

OUT_DIR.mkdir(parents=True, exist_ok=True)
tsv_out = OUT_DIR / "colleague_eval_small.tsv"
ltr_out = OUT_DIR / "colleague_eval_small.ltr"

missing = 0
written = 0

with open(tsv_out, "w") as ft, open(ltr_out, "w") as fl:
    ft.write(str(DOCKER_WAV_ROOT) + "\n")
    for _, row in df.iterrows():
        wav = WAV_DIR / (Path(row["path"]).stem + ".wav")
        if not wav.exists():
            missing += 1
            continue
        nframes = sf.info(str(wav)).frames
        gold = row["gold"]  # already normalized by colleague
        rel = wav.relative_to(ROOT / "data" / "wav")
        ft.write(f"{rel}\t{nframes}\n")
        fl.write(to_ltr(gold) + "\n")
        written += 1

print(f"Written: {written}  Missing: {missing}")
print(f"Manifest: {OUT_DIR.relative_to(ROOT)}/")
