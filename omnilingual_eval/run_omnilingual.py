#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run Omnilingual inference on wav files from a Common Voice TSV file.")
    parser.add_argument("--tsv", required=True, help="TSV file (CV format)")
    parser.add_argument("--audio_dir", required=True, help="Directory containing the wav files")
    parser.add_argument("--lang", default="qxp_Latn", help="Language code (Omnilingual format)")
    parser.add_argument("--output", default=None, help="Output TSV")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the pipeline")
    parser.add_argument("--save_every", type=int, default=50, help="Periodic backup every N files")
    parser.add_argument("--model_card", default="omniASR_LLM_7B_v2", help="Omnilingual model")
    parser.add_argument("--col_audio", default="path", help="Audio column name in CV TSV")
    parser.add_argument("--format", default="wav", help="Audio format: wav")
    return parser.parse_args()

def load_pipeline(model_card):
    try:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        print(f"Loading model {model_card}...")
        pipeline = ASRInferencePipeline(model_card=model_card)
        print("Model loaded.")
        return pipeline
    except ImportError:
        print("Install omnilingual-asr : pip install omnilingual-asr")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model : {e}")
        sys.exit(1)

def get_wav_path(audio_file, audio_dir):
    return audio_dir / f"{Path(audio_file).stem}.wav"

def main():
    args = parse_args()
    tsv_path = Path(args.tsv)
    audio_dir = Path(args.audio_dir)
    output_path = Path(args.output) if args.output else tsv_path.with_stem(tsv_path.stem + "_transcrit")
    temp_path = tsv_path.with_stem(tsv_path.stem + "_temp")  # fichier temporaire
    audio_file = args.col_audio
    
    if not tsv_path.exists() or not audio_dir.exists():
        print("TSV or directory not found.")
        sys.exit(1)

    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if audio_file not in df.columns:
        print(f"Column {audio_file} not in TSV.")
        sys.exit(1)
    if "transcription" not in df.columns:
        df["transcription"] = ""

    # Préparer la liste des fichiers à transcrire
    audio_paths = []
    indices = []
    missing = []

    for idx, row in df.iterrows():
        if args.format == "wav":
            path = get_wav_path(row[audio_file], audio_dir)
        else:
            path = audio_dir / Path(row[audio_file])
        
        if path.exists():
            audio_paths.append(str(path))
            indices.append(idx)
        else:
            df.at[idx, "transcription"] = "__FICHIER_MANQUANT__"
            missing.append(row[audio_file])

    if not audio_paths:
        print("There is no valid wav file.")
        df.to_csv(output_path, sep="\t", index=False)
        sys.exit(0)

    pipeline = load_pipeline(args.model_card)
    errors = []
    processed = 0

    # Transcription avec sauvegarde périodique
    for start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[start:start+args.batch_size]
        batch_indices = indices[start:start+args.batch_size]

        try:
            langs = [args.lang] * len(batch_paths)
            results = pipeline.transcribe(batch_paths, lang=langs, batch_size=args.batch_size)
            results = [r.strip() if isinstance(r, str) else str(r).strip() for r in results]

            for idx, text in zip(batch_indices, results):
                df.at[idx, "transcription"] = text
                filename = df.at[idx, audio_file]
                processed += 1
                print(f"[{processed}/{len(audio_paths)}] {filename} → {text[:80]}")

                # Sauvegarde périodique
                if processed % args.save_every == 0:
                    df.to_csv(temp_path, sep="\t", index=False)
                    print(f"→ Sauvegarde temporaire ({processed}) : {temp_path}")

        except Exception as e:
            print(f"Erreur batch : {e}")
            for idx in batch_indices:
                df.at[idx, "transcription"] = f"__ERREUR__: {e}"
                errors.append(df.at[idx, audio_file])

    # Sauvegarde finale
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nFinal file saved : {output_path}")

    if missing:
        print(f"{len(missing)} missing files : {missing}")
    if errors:
        print(f"{len(errors)} files with errors : {errors}")

    # Supprimer le fichier temporaire si tout est OK
    if temp_path.exists() and not errors:
        temp_path.unlink()
        print(f"Temp file deleted : {temp_path}")

if __name__ == "__main__":
    main()
