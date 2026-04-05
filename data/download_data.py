"""
Download Puno Quechua speech datasets from Mozilla Data Collective.

Datasets:
- Scripted Speech 25.0:   25,382 clips, 31h validated, 81 speakers, 2,070 sentences
- Spontaneous Speech 3.0: 7,286 clips,  5.2h validated, 110 speakers, natural responses

Requires: pip install datacollective python-dotenv
Auth:      copy .env.example to .env and fill in MDC_API_KEY
Usage:     python data/download_data.py [--output data/]
"""

import argparse
import os
import tarfile

from dotenv import load_dotenv
from datacollective import download_dataset

load_dotenv()

DATASETS = {
    "scripted": {
        "id": "cmn2cxfs801dvo107r5rkzmvx",
        "name": "Common Voice Scripted Speech 25.0 - Puno Quechua",
    },
    "spontaneous": {
        "id": "cmn1pujk200uno107g6el5r9y",
        "name": "Common Voice Spontaneous Speech 3.0 - Puno Quechua",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/", help="Directory to save datasets")
    args = parser.parse_args()

    for key, dataset in DATASETS.items():
        dest_dir = os.path.join(args.output, key)
        print(f"\nDownloading {dataset['name']}...")
        archive = download_dataset(dataset["id"], download_directory=dest_dir)
        print(f"  Extracting...")
        with tarfile.open(archive) as tar:
            tar.extractall(dest_dir)
        os.remove(archive)
        print(f"  Done → {dest_dir}")


if __name__ == "__main__":
    main()
