"""
Combine scripted and spontaneous fine-tune manifests into train_all.tsv / train_all.ltr.
Uses absolute paths in the TSV so both wav roots are supported in one file.
Filters out clips longer than 30s.

Usage (from project root):
  python training/scripts/combine_finetune_manifests.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FT_DIR = ROOT / "data/manifests/finetune/qxp"
MAX_FRAMES = 480000  # 30s at 16kHz


def process_split(tsv_path: Path, ltr_path: Path) -> tuple[list[str], list[str]]:
    """Read manifest and LTR, filter by max_frames, return (manifest_lines, ltr_lines)."""
    # Read manifest
    tsv_lines = tsv_path.read_text().splitlines()
    wav_root = Path(tsv_lines[0])
    manifest_entries = tsv_lines[1:]

    # Read LTR
    ltr_lines = [l for l in ltr_path.read_text().splitlines() if l.strip()]

    assert len(manifest_entries) == len(ltr_lines), \
        f"Mismatch in {tsv_path.name}: {len(manifest_entries)} manifest vs {len(ltr_lines)} LTR"

    # Filter and keep in sync
    filtered_manifest = []
    filtered_ltr = []
    skipped = 0

    for manifest_line, ltr_line in zip(manifest_entries, ltr_lines):
        if not manifest_line.strip():
            continue
        fname, nframes_str = manifest_line.split("\t")
        nframes = int(nframes_str)

        if nframes > MAX_FRAMES:
            skipped += 1
            continue

        # Resolve to absolute path
        abs_path = (wav_root / fname).resolve()
        filtered_manifest.append(f"{abs_path}\t{nframes}")
        filtered_ltr.append(ltr_line)

    if skipped:
        print(f"  {tsv_path.name}: skipped {skipped} samples > 30s")

    return filtered_manifest, filtered_ltr


# Process both splits
scripted_manifest, scripted_ltr = process_split(FT_DIR / "train.tsv", FT_DIR / "train.ltr")
spont_manifest, spont_ltr = process_split(FT_DIR / "train_spont.tsv", FT_DIR / "train_spont.ltr")

combined_manifest = scripted_manifest + spont_manifest
combined_ltr = scripted_ltr + spont_ltr

# Write combined files
FT_DIR.mkdir(parents=True, exist_ok=True)

with open(FT_DIR / "train_all.tsv", "w") as f:
    f.write("/\n")  # dummy root — all paths are absolute
    for line in combined_manifest:
        f.write(line + "\n")

(FT_DIR / "train_all.ltr").write_text("\n".join(combined_ltr) + "\n")

print(f"\nDone: {len(scripted_manifest)} scripted + {len(spont_manifest)} spontaneous = {len(combined_manifest)} total")
