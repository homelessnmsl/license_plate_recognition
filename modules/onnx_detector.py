"""ONNX Runtime YOLOv8 plate detector for deployment."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from config import DETECTOR, DetectorConfig
from modules.schemas import Detection


class ONNXModelUnavailableError(RuntimeError):
    """Raised when the exported ONNX detector cannot be loaded."""


class ONNXPlateDetector:
    """Dependency-light YOLOv8 detector backed by ONNX Runtime.

    The post-processing implements the standard YOLOv8 detection head:
    ``[cx, cy, width, height, class scores...]`` in letterboxed coordinates.
    It keeps the same ``detect`` interface as ``YOLOPlateDetector``.
    """

    def __init__(self, config: DetectorConfig = DETECTOR) -> None:
        self.config = config
        self._session = None
        self._input_name = ""
        self._input_height = int(config.imgsz)
        self._input_width = int(config.imgsz)

    def _providers(self) -> list[str]:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        requested = [
            item.strip()
            for item in os.getenv(
                "LPR_ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
            ).split(",")
            if item.strip()
        ]
        providers = [item for item in requested if item in available]
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    def _load(self) -> None:
        if self._session is not None:
            return
        weights = Path(self.config.weights)
        if not weights.is_file():
            raise ONNXModelUnavailableError(f"ONNX model not found: {weights}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ONNXModelUnavailableError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        try:
            self._session = ort.InferenceSession(
                str(weights), sess_options=options, providers=self._providers()
            )
        except Exception as exc:
            raise ONNXModelUnavailableError(
                f"Could not initialize ONNX model {weights}: {exc}"
            ) from exc

        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self._input_height = int(shape[2])
            self._input_width = int(shape[3])

    @staticmethod
    def _letterbox(
        image: np.ndarray, target_width: int, target_height: int
    ) -> tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        ratio = min(target_width / width, target_height / height)
        resized_width = max(1, int(round(width * ratio)))
        resized_height = max(1, int(round(height * ratio)))
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_x = (target_width - resized_width) / 2.0
        pad_y = (target_height - resized_height) / 2.0
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        canvas = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return canvas, ratio, pad_x, pad_y

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(scores)[::-1]
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            ix1 = np.maximum(x1[current], x1[remaining])
            iy1 = np.maximum(y1[current], y1[remaining])
            ix2 = np.minimum(x2[current], x2[remaining])
            iy2 = np.minimum(y2[current], y2[remaining])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            union = areas[current] + areas[remaining] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = remaining[iou <= iou_threshold]
        return keep

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._load()
        assert self._session is not None
        height, width = image.shape[:2]
        letterboxed, ratio, pad_x, pad_y = self._letterbox(
            image, self._input_width, self._input_height
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: blob})
        if not outputs:
            return []

        prediction = np.asarray(outputs[0])
        if prediction.ndim == 3:
            prediction = prediction[0]
        if prediction.ndim != 2:
            return []
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return []

        boxes_xywh = prediction[:, :4]
        class_scores = prediction[:, 4:]
        if class_scores.shape[1] == 0:
            return []
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        mask = confidences >= float(self.config.conf)
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        centers = boxes_xywh[:, :2]
        sizes = boxes_xywh[:, 2:4]
        corners = np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)
        corners[:, [0, 2]] = (corners[:, [0, 2]] - pad_x) / ratio
        corners[:, [1, 3]] = (corners[:, [1, 3]] - pad_y) / ratio
        corners[:, [0, 2]] = np.clip(corners[:, [0, 2]], 0, width - 1)
        corners[:, [1, 3]] = np.clip(corners[:, [1, 3]], 0, height - 1)

        keep = self._nms(corners, confidences, float(self.config.iou))
        detections: list[Detection] = []
        for index in keep[: int(self.config.max_det)]:
            x1, y1, x2, y2 = (int(round(value)) for value in corners[index])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidences[index]),
                    class_id=int(class_ids[index]),
                    class_name="license_plate",
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)
