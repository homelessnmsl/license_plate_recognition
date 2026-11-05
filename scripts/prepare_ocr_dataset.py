#!/usr/bin/env python3
"""Build a rectified OCR training set from the YOLO/CCPD dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.postprocess import clean_candidate, is_valid_plate
from modules.preprocessing import read_image, write_image
from modules.rectifier import rectify_plate
from modules.schemas import Detection
from scripts.prepare_ccpd import parse_ccpd_stem


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_category(path: Path) -> str:
    match = re.search(r"_(ccpd_[a-z]+)_\d+$", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def quality_reason(crop: np.ndarray) -> str | None:
    if crop.size == 0:
        return "empty"
    height, width = crop.shape[:2]
    if height < 24 or width < 50:
        return "too_small"
    aspect = width / float(height)
    if not 1.35 <= aspect <= 8.0:
        return "bad_aspect"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 12.0:
        return "too_dark"
    if float(gray.mean()) > 248.0:
        return "too_bright"
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 8.0:
        return "too_blurry"
    return None


def build_split(
    image_dir: Path,
    label_dir: Path,
    output_image_dir: Path,
    label_file: Path,
    seen_hashes: set[str],
    stats: Counter,
    category_counts: Counter,
) -> None:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        parsed = parse_ccpd_stem(image_path.stem)
        if parsed is None:
            stats["skip_unparsable"] += 1
            continue
        plate_text, _ = parsed
        plate_text = clean_candidate(plate_text)
        if not is_valid_plate(plate_text):
            stats["skip_invalid_text"] += 1
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            stats["skip_missing_label"] += 1
            continue
        parts = label_path.read_text(encoding="utf-8").strip().split()
        if len(parts) != 5:
            stats["skip_bad_label"] += 1
            continue

        digest = file_hash(image_path)
        if digest in seen_hashes:
            stats["skip_duplicate"] += 1
            continue
        seen_hashes.add(digest)

        image = read_image(image_path)
        height, width = image.shape[:2]
        cx, cy, bw, bh = map(float, parts[1:])
        x1 = int((cx - bw / 2.0) * width)
        y1 = int((cy - bh / 2.0) * height)
        x2 = int((cx + bw / 2.0) * width)
        y2 = int((cy + bh / 2.0) * height)
        detection = Detection((x1, y1, x2, y2), 1.0)
        crop, _, quad_found = rectify_plate(image, detection, margin=0.08)
        reason = quality_reason(crop)
        if reason:
            stats[reason] += 1
            continue

        output_name = f"{image_path.stem}.jpg"
        write_image(output_image_dir / output_name, crop)
        lines.append(f"images/{output_image_dir.name}/{output_name}\t{plate_text}")
        stats["kept"] += 1
        category = path_category(image_path)
        category_counts[category] += 1
        if quad_found:
            stats["rectified"] += 1

    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CCPD crops for PaddleOCR.")
    parser.add_argument("--yolo-root", type=Path, default=Path(r"E:\license_plate_data\yolo"))
    parser.add_argument("--output", type=Path, default=Path(r"E:\license_plate_data\ocr"))
    parser.add_argument("--test-root", type=Path, default=PROJECT_ROOT / "data" / "test")
    args = parser.parse_args()

    stats: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    (args.output / "images" / "train").mkdir(parents=True, exist_ok=True)
    (args.output / "images" / "val").mkdir(parents=True, exist_ok=True)
    (args.output / "labels").mkdir(parents=True, exist_ok=True)

    build_split(
        args.yolo_root / "images" / "train",
        args.yolo_root / "labels" / "train",
        args.output / "images" / "train",
        args.output / "labels" / "train.txt",
        seen_hashes,
        stats,
        category_counts,
    )
    build_split(
        args.yolo_root / "images" / "val",
        args.yolo_root / "labels" / "val",
        args.output / "images" / "val",
        args.output / "labels" / "val.txt",
        seen_hashes,
        stats,
        category_counts,
    )

    test_images = args.output / "images" / "test"
    test_images.mkdir(parents=True, exist_ok=True)
    test_lines: list[str] = []
    truth_path = args.test_root / "ground_truth.csv"
    if truth_path.is_file():
        with truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                image_rel = (row.get("image_path") or "").replace("\\", "/")
                image_path = (
                    PROJECT_ROOT / image_rel
                    if not Path(image_rel).is_absolute()
                    else Path(image_rel)
                )
                text = clean_candidate((row.get("plate_text") or "").strip())
                if not image_path.is_file() or not is_valid_plate(text):
                    stats["skip_test"] += 1
                    continue
                image = read_image(image_path)
                try:
                    x1 = int(float(row["x1"]))
                    y1 = int(float(row["y1"]))
                    x2 = int(float(row["x2"]))
                    y2 = int(float(row["y2"]))
                except (KeyError, ValueError):
                    stats["skip_test"] += 1
                    continue
                crop, _, quad_found = rectify_plate(
                    image, Detection((x1, y1, x2, y2), 1.0), margin=0.08
                )
                reason = quality_reason(crop)
                if reason:
                    stats[f"test_{reason}"] += 1
                    continue
                output_name = image_path.stem + ".jpg"
                write_image(test_images / output_name, crop)
                test_lines.append(f"images/test/{output_name}\t{text}")
                if quad_found:
                    stats["test_rectified"] += 1
    (args.output / "labels" / "test.txt").write_text(
        "\n".join(test_lines) + ("\n" if test_lines else ""), encoding="utf-8"
    )

    try:
        import paddleocr

        dict_source = Path(paddleocr.__file__).parent / "ppocr" / "utils" / "ppocr_keys_v1.txt"
        if dict_source.is_file():
            shutil.copy2(dict_source, args.output / "ppocr_keys_v1.txt")
            stats["dict_copied"] += 1
    except Exception:
        pass

    manifest = {
        "source": str(args.yolo_root),
        "output": str(args.output),
        "stats": dict(stats),
        "category_counts": dict(category_counts),
        "train_lines": len((args.output / "labels" / "train.txt").read_text(encoding="utf-8").splitlines()),
        "val_lines": len((args.output / "labels" / "val.txt").read_text(encoding="utf-8").splitlines()),
        "test_lines": len(test_lines),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())