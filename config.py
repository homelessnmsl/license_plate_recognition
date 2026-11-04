"""Runtime configuration for the YOLOv8n + PaddleOCR system.

Paths are resolved from this file, so commands work from any working directory.
Model weights are intentionally not committed to Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass(frozen=True)
class PathConfig:
    base: Path = BASE_DIR
    data: Path = DATA_DIR
    models: Path = MODELS_DIR
    test: Path = TEST_DIR
    output: Path = OUTPUT_DIR
    images: Path = OUTPUT_DIR / "images"
    json: Path = OUTPUT_DIR / "json"
    video: Path = OUTPUT_DIR / "video"


@dataclass(frozen=True)
class DetectorConfig:
    weights: Path = field(
        default_factory=lambda: Path(
            os.getenv("LPR_YOLO_WEIGHTS", str(MODELS_DIR / "yolov8n_plate.pt"))
        )
    )
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.50
    max_det: int = 20
    device: str | None = field(
        default_factory=lambda: os.getenv("LPR_DEVICE", "").strip() or None
    )


@dataclass(frozen=True)
class OCRConfig:
    lang: str = field(default_factory=lambda: os.getenv("LPR_OCR_LANG", "ch"))
    use_gpu: bool = field(
        default_factory=lambda: os.getenv("LPR_OCR_USE_GPU", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    min_score: float = 0.35
    use_angle_cls: bool = False
    model_dir: Path = MODELS_DIR / "paddleocr"


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    plate_margin: float = 0.08
    min_plate_length: int = 7
    max_plate_length: int = 8
    ocr_variants: Tuple[str, ...] = ("color", "clahe")
    frame_skip: int = 3
    log_level: str = field(
        default_factory=lambda: os.getenv("LPR_LOG_LEVEL", "INFO").upper()
    )


PATHS = PathConfig()
DETECTOR = DetectorConfig()
OCR = OCRConfig()
CONFIG = PipelineConfig()


def ensure_directories() -> None:
    for path in (
        PATHS.data,
        PATHS.models,
        PATHS.test,
        PATHS.output,
        PATHS.images,
        PATHS.json,
        PATHS.video,
        OCR.model_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)