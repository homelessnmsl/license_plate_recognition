"""Ultralytics YOLO plate detector wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DETECTOR, DetectorConfig
from modules.schemas import Detection


class ModelUnavailableError(RuntimeError):
    """Raised when model dependencies or weights are missing."""


class YOLOPlateDetector:
    """Lazy YOLO detector.

    The trained weights must contain a single plate class or a class whose name
    contains ``plate``. This prevents silently treating COCO car/person classes
    as license plates.
    """

    def __init__(self, config: DetectorConfig = DETECTOR) -> None:
        self.config = config
        self._model = None
        self._plate_class_ids: set[int] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.config.weights).is_file():
            raise ModelUnavailableError(
                f"YOLO weights not found: {self.config.weights}. "
                "Train or place yolov8n_plate.pt under data/models/."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError(
                "ultralytics is not installed. Install requirements first."
            ) from exc

        self._model = YOLO(str(self.config.weights))
        names = getattr(self._model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        plate_ids = {
            int(index)
            for index, name in names.items()
            if "plate" in str(name).lower() or "车牌" in str(name)
        }
        if not plate_ids and len(names) == 1:
            plate_ids = {0}
        if not plate_ids:
            raise ModelUnavailableError(
                "The YOLO weights do not expose a license-plate class. "
                f"Detected classes: {names}"
            )
        self._plate_class_ids = plate_ids

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._load()
        assert self._model is not None
        results = self._model.predict(
            source=image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            max_det=self.config.max_det,
            device=self.config.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(self._model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        image_height, image_width = image.shape[:2]
        detections: list[Detection] = []

        for box in boxes:
            class_id = int(box.cls[0].item())
            if self._plate_class_ids and class_id not in self._plate_class_ids:
                continue
            coords = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = (int(round(value)) for value in coords)
            x1, x2 = max(0, min(x1, image_width - 1)), max(0, min(x2, image_width - 1))
            y1, y2 = max(0, min(y1, image_height - 1)), max(0, min(y2, image_height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0].item()),
                    class_id=class_id,
                    class_name=str(names.get(class_id, "license_plate")),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)