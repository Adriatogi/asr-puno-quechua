import argparse
from pathlib import Path

import pandas as pd
from jiwer import wer, cer

def parse_args():
    parser = argparse.ArgumentParser(description="Compute WER from TSV file with gold and auto transcriptions.")
    parser.add_argument("tsv_path", help="TSV file")
    parser.add_argument("--output", default=None, help="Output TSV")
    return parser.parse_args()
    

args = parse_args()
tsv_path = Path(args.tsv_path)
output_path = Path(args.output) if args.output else tsv_path.with_stem(tsv_path.stem + "_scored")

df = pd.read_csv(tsv_path, sep='\t')

def remove_punct(sentence):
    return "".join([x.lower() for x in sentence if x not in ['?', '!', '¿', '¡', '.', ',']])
    
df = df.dropna(subset=['sentence', 'transcription'])

# Drop error rows before normalization (remove_punct lowercases, breaking the marker check)
error_mask = df['transcription'].astype(str).str.startswith(('__ERREUR__', '__FICHIER_MANQUANT__'))
if error_mask.any():
    print(f"Skipping {error_mask.sum()} error/missing rows")
    df = df[~error_mask]

df['gold'] = df['sentence'].apply(remove_punct)
df['transcription'] = df['transcription'].apply(remove_punct)

df.to_csv(output_path, sep='\t', index=False)

# Micro-WER/CER: total edits / total words across all utterances (matches Whisper/fairseq)
global_WER = wer(df["gold"].tolist(), df["transcription"].tolist())
global_CER = cer(df["gold"].tolist(), df["transcription"].tolist())

print(f"WER: {global_WER:.4f}  ({global_WER*100:.2f}%)")
print(f"CER: {global_CER:.4f}  ({global_CER*100:.2f}%)")
print(f"n utterances: {len(df)}")
