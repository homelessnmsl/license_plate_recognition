#!/usr/bin/env python3
"""Convert CCPD filenames into a YOLO detection dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import os
import random
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from config import BASE_DIR, MODELS_DIR
from modules.preprocessing import read_image

PROVINCES = "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
CHARACTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"


def parse_ccpd_stem(stem: str) -> tuple[str, tuple[int, int, int, int]] | None:
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        left_top, right_bottom = parts[2].split("_")

        def parse_point(value: str) -> tuple[int, int]:
            left, right = value.replace("&", ",").split(",")
            return int(left), int(right)

        x1, y1 = parse_point(left_top)
        x2, y2 = parse_point(right_bottom)
        code = [int(value) for value in parts[4].split("_")]
        plate_text = PROVINCES[code[0]] + "".join(CHARACTERS[index] for index in code[1:])
        return plate_text, (x1, y1, x2, y2)
    except (ValueError, IndexError):
        return None


def unique_destination_name(stem: str, source: Path, used: set[str]) -> str:
    candidate = stem
    if candidate in used:
        suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
        candidate = f"{stem}_{suffix}"
    used.add(candidate)
    return candidate


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CCPD for YOLO training.")
    parser.add_argument("--ccpd-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "data" / "yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    if not args.ccpd_dir.is_dir():
        raise FileNotFoundError(args.ccpd_dir)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    images = sorted(args.ccpd_dir.rglob("*.jpg"))
    if args.max_images > 0:
        images = images[: args.max_images]
    if len(images) < 2:
        raise ValueError("CCPD directory must contain at least two images")

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.Random(args.seed).shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio))
    splits = {"val": images[:val_count], "train": images[val_count:]}
    used: set[str] = set()
    ground_truth: list[dict] = []
    converted = skipped = 0
    converted_by_split: Counter[str] = Counter()

    for split, split_images in splits.items():
        for source in split_images:
            parsed = parse_ccpd_stem(source.stem)
            if parsed is None:
                skipped += 1
                continue
            plate_text, (x1, y1, x2, y2) = parsed
            try:
                image = read_image(source)
            except (FileNotFoundError, ValueError):
                skipped += 1
                continue
            height, width = image.shape[:2]
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            stem = unique_destination_name(source.stem, source, used)
            destination = args.output / "images" / split / f"{stem}.jpg"
            label_path = args.output / "labels" / split / f"{stem}.txt"
            link_or_copy(source, destination, args.mode)

            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            label_path.write_text(
                f"0 {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}\n",
                encoding="utf-8",
            )
            ground_truth.append(
                {
                    "image_path": str(destination),
                    "plate_text": plate_text,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            converted += 1
            converted_by_split[split] += 1

    dataset_yaml = BASE_DIR / "configs" / "plate_dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.output.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "license_plate"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    gt_path = args.output / "ground_truth.csv"
    with gt_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "plate_text", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(ground_truth)

    print(f"converted={converted}, skipped={skipped}")
    print(f"train={converted_by_split['train']}, val={converted_by_split['val']}")
    print(f"dataset_yaml={dataset_yaml}")
    print(f"ground_truth={gt_path}")
    print(f"suggested_weights={MODELS_DIR / 'yolov8n_plate.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())