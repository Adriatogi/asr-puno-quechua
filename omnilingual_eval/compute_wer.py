import argparse
import pandas as pd
from jiwer import wer, cer

def parse_args():
    parser = argparse.ArgumentParser(description="Compute WER from TSV file with gold and auto transcriptions.")
    parser.add_argument(tsv_path, required=True, help="TSV file")
    parser.add_argument("--output", default=None, help="Output TSV")
    return parser.parse_args()
    

args = parse_args()
tsv_path = Path(args.tsv_path)
output_path = Path(args.output) if args.output else tsv_path

df = pd.read_csv(tsv_path, sep='\t')

def remove_punct(sentence):
    return "".join([x.lower() for x in sentence if x not in ['?', '!', '¿', '¡', '.', ',']])
    
df['gold'] = df['sentence'].apply(remove_punct)
df = df.dropna()

df["wer"] = df.apply(lambda x: wer(x["gold"], x["transcription"]), axis=1)
df["cer"] = df.apply(lambda x: cer(x["gold"], x["transcription"]), axis=1)

df.to_csv(output_path, sep='\t', index=False)

global_WER = df["wer"].mean()
global_CER = df["cer"].mean()

print(f"WER global: {global_WER:.4f}")
print(f"CER global: {global_CER:.4f}")
