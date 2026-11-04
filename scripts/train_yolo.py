#!/usr/bin/env python3
"""Fine-tune YOLOv8n on the prepared plate dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, MODELS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8n plate detector.")
    parser.add_argument(
        "--data", type=Path, default=BASE_DIR / "configs" / "plate_dataset.yaml"
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}. Run scripts/prepare_ccpd.py first."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before training.") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        pretrained=True,
        amp=True,
        project=str(BASE_DIR / "runs" / "detect"),
        name="plate_yolov8n",
    )

    save_dir = Path(getattr(results, "save_dir", BASE_DIR / "runs" / "detect" / "plate_yolov8n"))
    best = save_dir / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "yolov8n_plate.pt"
    if best.is_file():
        shutil.copy2(best, target)
        print(f"Best weights copied to: {target}")
    else:
        print(f"Training completed but best.pt was not found under: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())