"""
Prepare Common Voice Puno Quechua (qxp) data for wav2vec 2.0 training.

Does three things:
  1. Convert MP3 → WAV (16kHz mono) for scripted and/or spontaneous audio
  2. Build pre-train manifests (unsupervised — all audio, no transcriptions needed)
  3. Build fine-tune manifests + .ltr transcription files + dict.ltr.txt

Manifest format (fairseq):
  Line 1:  /absolute/path/to/wav/root/dir
  Line 2+: filename.wav\tnum_frames

LTR format (one line per clip, characters space-separated, words separated by |):
  I r q i k u n a | p a y k u n a p u r a ...

Usage (from project root):
  python training/scripts/prepare_cv_qxp_data.py [options]
"""

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
import torchaudio
from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]

SCRIPTED_META  = ROOT / "data/scripted/cv-corpus-25.0-2026-03-09/qxp/validated.tsv"
SCRIPTED_TRAIN = ROOT / "data/scripted/cv-corpus-25.0-2026-03-09/qxp/train.tsv"
SCRIPTED_DEV   = ROOT / "data/scripted/cv-corpus-25.0-2026-03-09/qxp/dev.tsv"
SCRIPTED_TEST  = ROOT / "data/scripted/cv-corpus-25.0-2026-03-09/qxp/test.tsv"
SCRIPTED_CLIPS = ROOT / "data/scripted/cv-corpus-25.0-2026-03-09/qxp/clips"

SPONT_META     = ROOT / "data/spontaneous/sps-corpus-3.0-2026-03-09-qxp/ss-corpus-qxp.tsv"
SPONT_CLIPS    = ROOT / "data/spontaneous/sps-corpus-3.0-2026-03-09-qxp/audios"


# ── audio conversion ───────────────────────────────────────────────────────────

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = 16000):
    """Convert a single MP3 to 16kHz mono WAV using ffmpeg."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(sample_rate), "-ac", "1",
         "-sample_fmt", "s16", str(wav_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def convert_directory(mp3_dir: Path, wav_dir: Path, skip_existing: bool = True):
    """Convert all MP3s in mp3_dir to WAVs in wav_dir. Returns {stem: wav_path}."""
    mp3_files = sorted(mp3_dir.glob("*.mp3"))
    print(f"  Converting {len(mp3_files)} files from {mp3_dir.name}/...")
    mapping = {}
    for mp3 in tqdm(mp3_files, unit="file"):
        wav = wav_dir / (mp3.stem + ".wav")
        if not (skip_existing and wav.exists()):
            convert_mp3_to_wav(mp3, wav)
        mapping[mp3.stem] = wav
    return mapping


# ── manifest helpers ───────────────────────────────────────────────────────────

def get_num_frames(wav_path: Path) -> int:
    """Return number of samples in a WAV file."""
    info = torchaudio.info(str(wav_path))
    return info.num_frames


def write_manifest(tsv_path: Path, wav_root: Path, entries: list[tuple[str, int]]):
    """
    Write a fairseq manifest TSV.
    entries: list of (wav_filename, num_frames)
    """
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "w") as f:
        f.write(str(wav_root.resolve()) + "\n")
        for fname, nframes in entries:
            f.write(f"{fname}\t{nframes}\n")
    print(f"  Wrote {len(entries)} entries → {tsv_path.relative_to(ROOT)}")


# ── text helpers ───────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation except apostrophes."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text, flags=re.UNICODE)
    return text


def text_to_ltr(text: str) -> str:
    """
    Convert a transcript to space-separated letters with | between words.
    e.g. "Irqikuna paykuna" → "i r q i k u n a | p a y k u n a"
    """
    text = normalize_text(text)
    words = text.split()
    return " | ".join(" ".join(list(w)) for w in words)


# ── pretrain manifests ─────────────────────────────────────────────────────────

def build_pretrain_manifests(wav_scripted: Path, wav_spontaneous: Path,
                              manifest_dir: Path, splits: str):
    pretrain_dir = manifest_dir / "pretrain"

    # scripted — all clips (no labels needed)
    if splits in ("scripted", "both"):
        wav_files = sorted(wav_scripted.glob("*.wav"))
        entries = []
        print("Building qxp_scripted pretrain manifest...")
        for wav in tqdm(wav_files, unit="file"):
            entries.append((wav.name, get_num_frames(wav)))
        write_manifest(pretrain_dir / "qxp_scripted.tsv", wav_scripted, entries)

    # spontaneous — all clips
    if splits in ("spontaneous", "both"):
        wav_files = sorted(wav_spontaneous.glob("*.wav"))
        entries = []
        print("Building qxp_spontaneous pretrain manifest...")
        for wav in tqdm(wav_files, unit="file"):
            entries.append((wav.name, get_num_frames(wav)))
        write_manifest(pretrain_dir / "qxp_spontaneous.tsv", wav_spontaneous, entries)

    # validation manifest — scripted dev split
    print("Building qxp_valid pretrain manifest (scripted dev)...")
    df_dev = pd.read_csv(SCRIPTED_DEV, sep="\t")
    entries = []
    for _, row in tqdm(df_dev.iterrows(), total=len(df_dev), unit="clip"):
        stem = Path(row["path"]).stem
        wav = wav_scripted / (stem + ".wav")
        if wav.exists():
            entries.append((wav.name, get_num_frames(wav)))
    write_manifest(pretrain_dir / "qxp_valid.tsv", wav_scripted, entries)


# ── finetune manifests ─────────────────────────────────────────────────────────

def build_finetune_manifests(wav_scripted: Path, wav_spontaneous: Path,
                              manifest_dir: Path):
    ft_dir = manifest_dir / "finetune" / "qxp"
    ft_dir.mkdir(parents=True, exist_ok=True)

    # ── scripted splits (train / valid / test already defined in TSVs) ──
    split_files = {"train": SCRIPTED_TRAIN, "valid": SCRIPTED_DEV, "test": SCRIPTED_TEST}
    all_ltr_chars: Counter = Counter()

    for split, tsv_path in split_files.items():
        df = pd.read_csv(tsv_path, sep="\t")
        entries = []
        ltr_lines = []
        missing = 0
        for _, row in df.iterrows():
            stem = Path(row["path"]).stem
            wav = wav_scripted / (stem + ".wav")
            if not wav.exists():
                missing += 1
                continue
            transcript = str(row.get("sentence", "")).strip()
            if not transcript:
                missing += 1
                continue
            ltr = text_to_ltr(transcript)
            entries.append((wav.name, get_num_frames(wav)))
            ltr_lines.append(ltr)
            all_ltr_chars.update(ltr.replace(" ", "").replace("|", ""))

        write_manifest(ft_dir / f"{split}.tsv", wav_scripted, entries)
        ltr_path = ft_dir / f"{split}.ltr"
        ltr_path.write_text("\n".join(ltr_lines) + "\n")
        print(f"  Wrote {len(ltr_lines)} transcripts → {ltr_path.relative_to(ROOT)}"
              + (f"  ({missing} skipped)" if missing else ""))

    # ── spontaneous validated clips (use split column) ──
    df_spont = pd.read_csv(SPONT_META, sep="\t")
    df_validated = df_spont[df_spont["transcription"].notna()].copy()

    for split in ("train", "valid", "test"):
        subset = df_validated[df_validated["split"] == split]
        entries = []
        ltr_lines = []
        missing = 0
        for _, row in subset.iterrows():
            stem = Path(row["audio_file"]).stem
            wav = wav_spontaneous / (stem + ".wav")
            if not wav.exists():
                missing += 1
                continue
            ltr = text_to_ltr(str(row["transcription"]))
            entries.append((wav.name, get_num_frames(wav)))
            ltr_lines.append(ltr)
            all_ltr_chars.update(ltr.replace(" ", "").replace("|", ""))

        if entries:
            # Append spontaneous to the scripted split files
            spont_tsv = ft_dir / f"{split}_spont.tsv"
            spont_ltr = ft_dir / f"{split}_spont.ltr"
            write_manifest(spont_tsv, wav_spontaneous, entries)
            spont_ltr.write_text("\n".join(ltr_lines) + "\n")
            print(f"  Wrote {len(ltr_lines)} spontaneous transcripts → "
                  f"{spont_ltr.relative_to(ROOT)}"
                  + (f"  ({missing} skipped)" if missing else ""))

    # ── dict.ltr.txt ──
    dict_path = ft_dir / "dict.ltr.txt"
    with open(dict_path, "w") as f:
        for char, count in sorted(all_ltr_chars.items(), key=lambda x: -x[1]):
            if char.strip():
                f.write(f"{char} {count}\n")
    print(f"  Wrote {len(all_ltr_chars)} letter types → {dict_path.relative_to(ROOT)}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice Puno Quechua data for wav2vec 2.0 training."
    )
    parser.add_argument("--audio-dir-scripted",  default=str(SCRIPTED_CLIPS))
    parser.add_argument("--audio-dir-spontaneous", default=str(SPONT_CLIPS))
    parser.add_argument("--wav-dir",    default=str(ROOT / "data/wav"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "data/manifests"))
    parser.add_argument("--splits",     default="both",
                        choices=["scripted", "spontaneous", "both"],
                        help="Which datasets to include in CPT pretrain manifests")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip MP3→WAV conversion (if already done)")
    args = parser.parse_args()

    wav_dir      = Path(args.wav_dir)
    manifest_dir = Path(args.manifest_dir)
    wav_scripted    = wav_dir / "scripted"
    wav_spontaneous = wav_dir / "spontaneous"

    # 1. Convert audio
    if not args.skip_conversion:
        print("\n=== Converting audio MP3 → WAV (16kHz mono) ===")
        if args.splits in ("scripted", "both"):
            convert_directory(Path(args.audio_dir_scripted), wav_scripted)
        if args.splits in ("spontaneous", "both"):
            convert_directory(Path(args.audio_dir_spontaneous), wav_spontaneous)
        # Always convert scripted for finetune (needs labels)
        if args.splits == "spontaneous":
            print("  (also converting scripted for fine-tune manifests)")
            convert_directory(Path(args.audio_dir_scripted), wav_scripted)
    else:
        print("Skipping audio conversion.")

    # 2. Pre-train manifests
    print("\n=== Building pre-train manifests ===")
    build_pretrain_manifests(wav_scripted, wav_spontaneous, manifest_dir, args.splits)

    # 3. Fine-tune manifests
    print("\n=== Building fine-tune manifests ===")
    build_finetune_manifests(wav_scripted, wav_spontaneous, manifest_dir)

    print("\nDone.")
    print(f"  Pre-train manifests : {manifest_dir}/pretrain/")
    print(f"  Fine-tune manifests : {manifest_dir}/finetune/qxp/")


if __name__ == "__main__":
    main()
