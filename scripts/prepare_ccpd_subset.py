#!/usr/bin/env python3
"""Download and sample a reproducible CCPD subset from Hugging Face.

The selected 30k repository keeps original CCPD filenames, which is essential:
filename annotations contain both the plate bounding box and text. A stratified
subset is extracted for YOLO training, and a disjoint 30-image test set is
written with exact labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_ccpd import parse_ccpd_stem

MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{MIRROR}/datasets/zenitsu09/ccpd-subset-30k/resolve/main/ccpd_subset_30k.zip"
TEST_URL = f"{MIRROR}/datasets/zenitsu09/ccpd-100k-yolo/resolve/main/ccpd_test_1k.zip"

TRAIN_TARGETS = {
    "ccpd_base": 4000,
    "ccpd_challenge": 1500,
    "ccpd_tilt": 1500,
    "ccpd_green": 1000,
    "ccpd_blur": 900,
    "ccpd_fn": 900,
    "ccpd_rotate": 500,
    "ccpd_db": 700,
    "ccpd_weather": 700,
    "ccpd_np": 400,
}
TEST_TARGETS = {
    "ccpd_base": 6,
    "ccpd_challenge": 3,
    "ccpd_tilt": 3,
    "ccpd_blur": 3,
    "ccpd_fn": 3,
    "ccpd_green": 3,
    "ccpd_db": 3,
    "ccpd_rotate": 3,
    "ccpd_weather": 3,
}
CATEGORY_PATTERN = re.compile(r"_(ccpd_[a-z]+)_\d+\.jpg$", re.IGNORECASE)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"skip download: {destination} ({destination.stat().st_size / 1024**2:.1f} MB)")
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    mode = "ab" if offset else "wb"
    print(f"download: {url}")
    with requests.get(url, stream=True, headers=headers, timeout=(30, 180)) as response:
        if response.status_code == 416 and offset:
            partial.replace(destination)
            return
        response.raise_for_status()
        if offset and response.status_code != 206:
            mode = "wb"
            offset = 0
        downloaded = offset
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded // (200 * 1024**2) != (downloaded - len(chunk)) // (200 * 1024**2):
                        print(f"  downloaded {downloaded / 1024**2:.0f} MB")
    partial.replace(destination)
    print(f"saved: {destination} ({destination.stat().st_size / 1024**2:.1f} MB)")


def group_images(zip_path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".jpg"):
                match = CATEGORY_PATTERN.search(name)
                if match:
                    groups[match.group(1).lower()].append(name)
    return groups


def extract_train_subset(zip_path: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    groups = group_images(zip_path)
    rng = random.Random(seed)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for category, limit in TRAIN_TARGETS.items():
        candidates = groups.get(category, [])
        picks = rng.sample(candidates, min(limit, len(candidates)))
        selected.extend(picks)
        counts[category] = len(picks)
    rng.shuffle(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for index, name in enumerate(selected, start=1):
            target = output_dir / Path(name).name
            if not target.exists():
                with archive.open(name) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            if index % 2000 == 0:
                print(f"  extracted {index}/{len(selected)}")
    print(f"train subset: {len(selected)} images")
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_test_subset(
    zip_path: Path,
    test_dir: Path,
    train_dir: Path,
    seed: int = 2026,
) -> list[dict]:
    groups = group_images(zip_path)
    train_suffixes = {
        match.group(0)
        for path in train_dir.glob("*.jpg")
        if (match := CATEGORY_PATTERN.search(path.name))
    }
    print("hashing training images for content-level test isolation...")
    train_hashes = {sha256_file(path) for path in train_dir.glob("*.jpg")}
    rng = random.Random(seed)
    selected: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        candidate_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for names in groups.values()
            for name in names
        }
    for category, count in TEST_TARGETS.items():
        candidates = [
            name
            for name in groups.get(category, [])
            if not ((match := CATEGORY_PATTERN.search(name)) and match.group(0) in train_suffixes)
            and candidate_hashes[name] not in train_hashes
        ]
        if len(candidates) < count:
            raise ValueError(f"Not enough disjoint test images for {category}")
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)

    test_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as archive:
        for index, name in enumerate(selected, start=1):
            parsed = parse_ccpd_stem(Path(name).stem)
            if parsed is None:
                raise ValueError(f"Cannot parse CCPD annotation from {name}")
            plate_text, (x1, y1, x2, y2) = parsed
            output_name = f"ccpd_test_{index:03d}.jpg"
            with archive.open(name) as source, (test_dir / output_name).open("wb") as destination:
                destination.write(source.read())
            image_path = test_dir / output_name
            try:
                csv_path = image_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                csv_path = str(image_path)
            rows.append(
                {
                    "image_path": csv_path,
                    "plate_text": plate_text,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    with (test_dir / "ground_truth.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "plate_text", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"test subset: {len(rows)} disjoint images with exact labels")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and sample CCPD.")
    parser.add_argument("--root", type=Path, default=Path(r"E:\license_plate_data"))
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    downloads = args.root / "downloads"
    train_zip = downloads / "ccpd_subset_30k.zip"
    test_zip = downloads / "ccpd_test_1k.zip"
    if not args.skip_download:
        download(TRAIN_URL, train_zip)
        download(TEST_URL, test_zip)

    raw_dir = args.root / "raw" / "ccpd_mixed_12k"
    counts = extract_train_subset(train_zip, raw_dir, seed=args.seed)
    print("category counts:", dict(counts))

    if not args.skip_test:
        extract_test_subset(test_zip, args.test_dir, raw_dir, seed=2026)

    print("Next command:")
    print(
        "python scripts/prepare_ccpd.py "
        f"--ccpd-dir {raw_dir} --output {args.root / 'yolo'} --val-ratio 0.15"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())