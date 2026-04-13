"""
Build fairseq fine-tuning manifests from colleague's data/splits_joh/ TSVs.

Outputs to data/manifests/finetune/qxp_v2/:
  train.tsv + train.ltr       Config 1: validated scripted + validated spontaneous ×3
  train_silver.tsv + train_silver.ltr  Config 2: validated scripted + silver spontaneous
  valid.tsv + valid.ltr       scripted dev + spontaneous dev
  test.tsv + test.ltr         scripted test (2,299 clips)
  test_spont.tsv + test_spont.ltr  spontaneous test (371 clips)
  dict.ltr.txt                character vocab

Usage (from project root):
  python training/scripts/build_finetune_manifests.py
"""

import random
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_SPLIT = ROOT / "data" / "splits_joh"
WAV_SCRIPTED = ROOT / "data" / "wav" / "scripted"
WAV_SPONTANEOUS = ROOT / "data" / "wav" / "spontaneous"
OUT_DIR = ROOT / "data" / "manifests" / "finetune" / "qxp_v2"
# Paths written to manifests must work inside Docker, where the project root
# is bind-mounted at /workspace. Use /workspace/data/wav as manifest root
# and relative paths (scripted/foo.wav, spontaneous/bar.wav).
DOCKER_WAV_ROOT = Path("/workspace/data/wav")
UPSAMPLE_FACTOR = 3  # matches train_whisper_validated.py
MAX_FRAMES = 480_000  # 30s at 16kHz — skip clips longer than this


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase and strip punctuation — matches compute_wer.py."""
    text = text.lower()
    return "".join(c for c in text if c not in "?!¿¡.,").strip()


def to_ltr(text: str) -> str:
    """Convert normalized sentence to fairseq LTR format (space → '|')."""
    return " ".join("|" if c == " " else c for c in text)


def load_split(tsv_path: Path, wav_dir: Path, sentence_col: str = "sentence") -> list[dict]:
    """
    Read a TSV, resolve each 'path' column entry to a WAV file, return list of dicts:
      {wav_path: Path, nframes: int, sentence: str, ltr: str}
    Rows with missing WAVs are skipped (shouldn't happen — all WAVs verified present).
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=["path", sentence_col])
    rows = []
    missing = 0
    skipped_length = 0
    for _, row in df.iterrows():
        wav = wav_dir / Path(row["path"]).with_suffix(".wav").name
        if not wav.exists():
            missing += 1
            continue
        nframes = sf.info(str(wav)).frames
        if nframes > MAX_FRAMES:
            skipped_length += 1
            continue
        sentence = normalize(str(row[sentence_col]))
        rows.append({
            "wav_path": wav,
            "nframes": nframes,
            "sentence": sentence,
            "ltr": to_ltr(sentence),
        })
    if missing:
        print(f"  WARNING: {missing} missing WAVs in {tsv_path.name}")
    if skipped_length:
        print(f"  Skipped {skipped_length} clips over {MAX_FRAMES} frames in {tsv_path.name}")
    return rows


def write_manifest(rows: list[dict], tsv_out: Path, ltr_out: Path) -> None:
    with open(tsv_out, "w") as ft, open(ltr_out, "w") as fl:
        ft.write(str(DOCKER_WAV_ROOT) + "\n")
        for r in rows:
            # Relative to data/wav/ so Docker bind mount resolves correctly
            rel = r["wav_path"].relative_to(ROOT / "data" / "wav")
            ft.write(f"{rel}\t{r['nframes']}\n")
            fl.write(r["ltr"] + "\n")
    print(f"  {tsv_out.name}: {len(rows)} rows")


def build_dict(all_ltr_rows: list[str]) -> Counter:
    counts: Counter = Counter()
    for line in all_ltr_rows:
        for tok in line.split():
            counts[tok] += 1
    return counts


# ---------------------------------------------------------------------------
# load all splits
# ---------------------------------------------------------------------------

print("Loading scripted splits...")
scripted_train = load_split(DATA_SPLIT / "validated_scripted" / "train.tsv",     WAV_SCRIPTED)
scripted_dev   = load_split(DATA_SPLIT / "validated_scripted" / "dev.tsv",       WAV_SCRIPTED)
scripted_test  = load_split(DATA_SPLIT / "validated_scripted" / "test.tsv",      WAV_SCRIPTED)
print(f"  train={len(scripted_train)}  dev={len(scripted_dev)}  test={len(scripted_test)}")

print("Loading validated spontaneous splits...")
spont_train = load_split(
    DATA_SPLIT / "validated_spontaneous" / "train_spontaneous.tsv",
    WAV_SPONTANEOUS,
    sentence_col="sentence",
)
spont_dev = load_split(
    DATA_SPLIT / "validated_spontaneous" / "dev_spontaneous.tsv",
    WAV_SPONTANEOUS,
    sentence_col="sentence",
)
spont_test = load_split(
    DATA_SPLIT / "validated_spontaneous" / "test_spontaneous.tsv",
    WAV_SPONTANEOUS,
    sentence_col="sentence",
)
print(f"  train={len(spont_train)}  dev={len(spont_dev)}  test={len(spont_test)}")

print("Loading silver spontaneous train...")
silver_train = load_split(
    DATA_SPLIT / "silver_spontaneous" / "train.tsv",
    WAV_SPONTANEOUS,
    sentence_col="sentence",
)
print(f"  train={len(silver_train)}")

# ---------------------------------------------------------------------------
# build Config 1 train: validated scripted + validated spontaneous ×3
# ---------------------------------------------------------------------------

rng = random.Random(SEED)

spont_upsampled = spont_train * UPSAMPLE_FACTOR
train_validated = scripted_train + spont_upsampled
rng.shuffle(train_validated)
print(f"\nConfig 1 train: {len(scripted_train)} scripted + {len(spont_upsampled)} spont×{UPSAMPLE_FACTOR} = {len(train_validated)}")

# ---------------------------------------------------------------------------
# build Config 2 train: validated scripted + silver spontaneous (no upsample)
# ---------------------------------------------------------------------------

train_silver = scripted_train + silver_train
rng2 = random.Random(SEED)
rng2.shuffle(train_silver)
print(f"Config 2 train: {len(scripted_train)} scripted + {len(silver_train)} silver = {len(train_silver)}")

# ---------------------------------------------------------------------------
# build valid: scripted dev + spontaneous dev (combined, no upsample)
# ---------------------------------------------------------------------------

valid = scripted_dev + spont_dev
print(f"Valid: {len(scripted_dev)} scripted + {len(spont_dev)} spont = {len(valid)}")

# ---------------------------------------------------------------------------
# write manifests
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nWriting to {OUT_DIR.relative_to(ROOT)}/")

write_manifest(train_validated, OUT_DIR / "train.tsv",        OUT_DIR / "train.ltr")
write_manifest(train_silver,    OUT_DIR / "train_silver.tsv", OUT_DIR / "train_silver.ltr")
write_manifest(valid,           OUT_DIR / "valid.tsv",        OUT_DIR / "valid.ltr")
write_manifest(scripted_test,   OUT_DIR / "test.tsv",         OUT_DIR / "test.ltr")
write_manifest(spont_test,      OUT_DIR / "test_spont.tsv",   OUT_DIR / "test_spont.ltr")

# ---------------------------------------------------------------------------
# build dict.ltr.txt from all training LTR lines
# ---------------------------------------------------------------------------

all_train_ltr = (
    [r["ltr"] for r in train_validated]
    + [r["ltr"] for r in silver_train]  # include silver chars too
)
char_counts = build_dict(all_train_ltr)
dict_path = OUT_DIR / "dict.ltr.txt"
with open(dict_path, "w") as f:
    for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        f.write(f"{char} {count}\n")
print(f"  dict.ltr.txt: {len(char_counts)} characters")

print("\nDone.")
