"""
Parse fairseq infer output and compute micro-WER + micro-CER.

The old infer.py writes:
  hypo.word-checkpoint_best.pt-<subset>.txt
  ref.word-checkpoint_best.pt-<subset>.txt
  hypo.units-checkpoint_best.pt-<subset>.txt
  ref.units-checkpoint_best.pt-<subset>.txt

Each line format:  <text> (None-<id>)
  - word files: words separated by <unk> (word boundary token)
  - units files: space-separated chars, <unk> as word boundary

Usage (from project root, conda activate asr-puno):
  python eval/analyze_results.py [--results-dir results]
"""

import argparse
import csv
import re
from pathlib import Path

import jiwer

MODELS = ["ft_cpt_validated", "ft_cpt_silver", "ft_xlsr_validated", "ft_xlsr_silver"]
SUBSETS = ["test", "test_spont"]


def parse_lines(path: Path):
    """
    Read a hypo/ref file and return {id: text} dict.
    Line format: <text> (None-<id>)
    """
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"\(None-(\d+)\)$", line)
            if not m:
                continue
            idx = int(m.group(1))
            text = line[: m.start()].strip()
            entries[idx] = text
    return entries


def lines_to_words(entries: dict) -> list[str]:
    """Convert word-file entries to word sequences (replace <unk> with space)."""
    ids = sorted(entries)
    return [entries[i].replace("<unk>", " ").strip() for i in ids]


def lines_to_chars(entries: dict) -> list[str]:
    """
    Convert units-file entries to character strings for CER.
    Units are space-separated chars; <unk> is the word boundary.
    Join chars, treating <unk> as space.
    """
    ids = sorted(entries)
    result = []
    for i in ids:
        text = entries[i]
        # Tokens are space-separated; <unk> = word boundary → space
        tokens = text.split()
        chars = "".join(" " if t == "<unk>" else t for t in tokens)
        result.append(chars.strip())
    return result


def find_files(results_dir: Path, model: str, subset: str):
    base = results_dir / model / subset
    hypo_word = list(base.glob(f"hypo.word-*{subset}*.txt"))
    ref_word  = list(base.glob(f"ref.word-*{subset}*.txt"))
    hypo_unit = list(base.glob(f"hypo.units-*{subset}*.txt"))
    ref_unit  = list(base.glob(f"ref.units-*{subset}*.txt"))
    return (
        hypo_word[0] if hypo_word else None,
        ref_word[0]  if ref_word  else None,
        hypo_unit[0] if hypo_unit else None,
        ref_unit[0]  if ref_unit  else None,
    )


def main(results_dir: Path, models: list = MODELS, subsets: list = SUBSETS):
    summary = []

    for model in models:
        for subset in subsets:
            hw, rw, hu, ru = find_files(results_dir, model, subset)

            if not hw or not rw:
                print(f"  MISSING word files: {model}/{subset}")
                continue

            hypo_entries = parse_lines(hw)
            ref_entries  = parse_lines(rw)
            ids = sorted(set(hypo_entries) & set(ref_entries))
            if not ids:
                print(f"  NO MATCHING IDs: {model}/{subset}")
                continue

            hypo_words = lines_to_words({i: hypo_entries[i] for i in ids})
            ref_words  = lines_to_words({i: ref_entries[i]  for i in ids})
            wer = jiwer.wer(ref_words, hypo_words) * 100

            # CER from units files if available
            cer_str = "—"
            if hu and ru:
                hypo_unit_entries = parse_lines(hu)
                ref_unit_entries  = parse_lines(ru)
                ids_u = sorted(set(hypo_unit_entries) & set(ref_unit_entries))
                hypo_chars = lines_to_chars({i: hypo_unit_entries[i] for i in ids_u})
                ref_chars  = lines_to_chars({i: ref_unit_entries[i]  for i in ids_u})
                cer = jiwer.cer(ref_chars, hypo_chars) * 100
                cer_str = f"{cer:.2f}%"

            n = len(ids)
            print(f"  {model:25s} {subset:12s}  WER={wer:6.2f}%  CER={cer_str:8s}  n={n}")
            summary.append({
                "model": model,
                "subset": subset,
                "wer": f"{wer:.2f}",
                "cer": cer_str.replace("%", "").strip(),
                "n_utterances": n,
            })

    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "subset", "wer", "cer", "n_utterances"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--subsets", nargs="+", default=SUBSETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    print(f"Parsing results from {args.results_dir}/\n")
    main(args.results_dir, models=args.models, subsets=args.subsets)
