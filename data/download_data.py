"""
Download Puno Quechua speech datasets from Mozilla Data Collective.

Datasets:
- Scripted Speech 25.0:   25,382 clips, 31h validated, 81 speakers, 2,070 sentences
- Spontaneous Speech 3.0: 7,286 clips,  5.2h validated, 110 speakers, natural responses

Requires: pip install requests python-dotenv tqdm
Auth:      copy .env.example to .env and fill in MDC_API_KEY
Usage:     python data/download_data.py [--output data/]
"""

import argparse
import os
import tarfile

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_BASE = "https://datacollective.mozillafoundation.org/api/datasets"

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


def get_download_url(dataset_id, api_key):
    resp = requests.post(
        f"{API_BASE}/{dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["downloadUrl"]


def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/", help="Directory to save datasets")
    args = parser.parse_args()

    api_key = os.getenv("MDC_API_KEY")
    if not api_key:
        raise EnvironmentError("MDC_API_KEY not set. Copy .env.example to .env and fill it in.")

    for key, dataset in DATASETS.items():
        dest_dir = os.path.join(args.output, key)
        archive = os.path.join(args.output, f"{key}.tar.gz")
        os.makedirs(args.output, exist_ok=True)

        print(f"\nDownloading {dataset['name']}...")
        url = get_download_url(dataset["id"], api_key)
        download_file(url, archive)

        print("  Extracting...")
        with tarfile.open(archive) as tar:
            tar.extractall(dest_dir)
        os.remove(archive)
        print(f"  Done → {dest_dir}")


if __name__ == "__main__":
    main()
