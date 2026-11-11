#!/usr/bin/env python3
"""Package the trained detector and OCR model for deployment.

YOLOv8 is exported to ONNX for runtime-independent inference. The fine-tuned
PaddleOCR recognition model is already stored in Paddle Inference format and is
copied together with its configuration and label dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO = ROOT / "data" / "models" / "yolov8n_plate.pt"
DEFAULT_OCR = Path(r"E:\license_plate_data\ocr_train_output\inference")
DEFAULT_OUTPUT = ROOT / "deploy" / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_yolo(weights: Path, output_dir: Path, imgsz: int, device: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=12,
            dynamic=False,
            simplify=False,
            half=False,
            device=device or None,
        )
    ).resolve()
    target = output_dir / "yolov8n_plate.onnx"
    if exported != target:
        shutil.copy2(exported, target)
    shutil.copy2(weights, output_dir / "yolov8n_plate.pt")
    return target


def copy_ocr_model(source: Path, output_dir: Path) -> Path:
    required = ("inference.pdmodel", "inference.pdiparams")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"OCR inference model is incomplete: {missing}")
    target = output_dir / "ocr_rec"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and (
            item.name.startswith("inference.")
            or item.name in {"inference.yml", "config.yml"}
        ):
            shutil.copy2(item, target / item.name)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and package plate models.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--ocr-model-dir", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "YOLOv8n detector + PP-OCRv4 recognition",
        "input_size": args.imgsz,
    }

    if not args.skip_yolo:
        if not args.yolo_weights.is_file():
            raise FileNotFoundError(args.yolo_weights)
        print(f"Exporting YOLO ONNX from {args.yolo_weights} ...")
        onnx_path = export_yolo(
            args.yolo_weights.resolve(), args.output_dir, args.imgsz, args.device
        )
        manifest["yolo"] = {
            "source": str(args.yolo_weights.resolve()),
            "onnx": onnx_path.name,
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_sha256": sha256(onnx_path),
            "pt": "yolov8n_plate.pt",
            "classes": ["license_plate"],
            "opset": 12,
            "dynamic": False,
        }
        print(f"YOLO ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MiB)")

    if not args.skip_ocr:
        if not args.ocr_model_dir.is_dir():
            raise FileNotFoundError(args.ocr_model_dir)
        print(f"Packaging PaddleOCR model from {args.ocr_model_dir} ...")
        ocr_dir = copy_ocr_model(args.ocr_model_dir.resolve(), args.output_dir)
        manifest["ocr"] = {
            "source": str(args.ocr_model_dir.resolve()),
            "directory": ocr_dir.name,
            "format": "Paddle Inference",
            "files": {
                item.name: {"bytes": item.stat().st_size, "sha256": sha256(item)}
                for item in sorted(ocr_dir.iterdir())
                if item.is_file()
            },
        }
        print(f"OCR model: {ocr_dir}")

    evaluation_onnx = ROOT / "outputs" / "json" / "evaluation_onnx.json"
    evaluation = (
        evaluation_onnx
        if evaluation_onnx.is_file()
        else ROOT / "outputs" / "json" / "evaluation.json"
    )
    if evaluation.is_file():
        try:
            data = json.loads(evaluation.read_text(encoding="utf-8"))
            manifest["validation"] = {"report": evaluation.name}
            manifest["validation"].update(
                {
                    key: data.get(key)
                    for key in (
                        "total",
                        "recognition_accuracy",
                        "character_accuracy",
                        "mean_processing_time_ms",
                    )
                }
            )
        except (OSError, ValueError):
            pass

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
