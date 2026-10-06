import argparse
import hashlib
from pathlib import Path

import requests


BASE_URL = "https://zenodo.org/records/10203108/files"
FILES = {
    "calendar.csv": "3ffeab2991b0c8e861d008b39ea4c95c",
    "sales_train_evaluation.csv": "b806dfc9f30a745102b708c09951f6aa",
    "sell_prices.csv": "08c591caa99e55daf3e0ccac913f7c85",
}


def checksum(path):
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = directory / name
        if target.exists() and checksum(target) == expected:
            print(f"{name}: already downloaded")
            continue
        temporary = target.with_suffix(".part")
        print(f"Downloading {name}", flush=True)
        with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for block in response.iter_content(1024 * 1024):
                    file.write(block)
        if checksum(temporary) != expected:
            temporary.unlink()
            raise ValueError(f"Checksum mismatch for {name}")
        temporary.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    download(args.output)
