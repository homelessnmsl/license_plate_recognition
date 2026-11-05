"""End-to-end YOLO detection + PaddleOCR recognition pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from config import CONFIG, PipelineConfig
from modules.detector import YOLOPlateDetector
from modules.postprocess import make_candidate, rank_candidates
from modules.preprocessing import read_image
from modules.recognizer import PaddlePlateRecognizer
from modules.rectifier import classify_plate_color, crop_detection, rectify_plate
from modules.schemas import Detection, OCRCandidate, PlateResult


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter)


class LicensePlatePipeline:
    """Two-stage detector/recognizer pipeline with dependency injection.

    The detector and recognizer are injectable so the orchestration can be
    unit-tested without loading PyTorch or PaddlePaddle.
    """

    def __init__(
        self,
        detector: object | None = None,
        recognizer: object | None = None,
        config: PipelineConfig = CONFIG,
    ) -> None:
        self.config = config
        self.detector = detector or YOLOPlateDetector(config.detector)
        self.recognizer = recognizer or PaddlePlateRecognizer(config.ocr)

    def process_array(self, image: np.ndarray, source: str = "<array>") -> list[PlateResult]:
        start = time.perf_counter()
        detections: list[Detection] = self.detector.detect(image)  # type: ignore[attr-defined]
        results: list[PlateResult] = []

        for detection in detections:
            plate_crop, plate_color, quad_found = rectify_plate(
                image, detection, margin=self.config.plate_margin
            )
            original_crop = crop_detection(
                image, detection, margin=self.config.plate_margin
            )
            if plate_color == "unknown":
                plate_color = classify_plate_color(original_crop)

            readings: list[OCRCandidate] = []
            for crop_source, crop in (
                ("rectified", plate_crop),
                ("detection", original_crop),
            ):
                if crop.size == 0:
                    continue
                for item in self.recognizer.recognize(crop):  # type: ignore[attr-defined]
                    readings.append(
                        OCRCandidate(
                            text=item.text,
                            confidence=item.confidence,
                            raw_text=item.raw_text,
                            variant=f"{crop_source}:{item.variant}",
                        )
                    )
            formatted = rank_candidates(
                [
                    make_candidate(item.text, item.confidence, item.variant)
                    for item in readings
                ]
            )

            needs_full_ocr = (
                not formatted
                or not formatted[0].valid
                or formatted[0].confidence < 0.92
                or (plate_color == "green" and len(formatted[0].text) != 8)
            )
            if needs_full_ocr and hasattr(self.recognizer, "recognize_full"):
                for crop_source, crop in (
                    ("rectified", plate_crop),
                    ("detection", original_crop),
                ):
                    if crop.size == 0:
                        continue
                    for item in self.recognizer.recognize_full(crop):  # type: ignore[attr-defined]
                        readings.append(
                            OCRCandidate(
                                text=item.text,
                                confidence=item.confidence,
                                raw_text=item.raw_text,
                                variant=f"{crop_source}:{item.variant}",
                            )
                        )
                formatted = rank_candidates(
                    [
                        make_candidate(item.text, item.confidence, item.variant)
                        for item in readings
                    ]
                )

            if not formatted:
                results.append(
                    PlateResult(
                        image_path=source,
                        bbox=detection.bbox,
                        detection_confidence=detection.confidence,
                        plate_text="",
                        ocr_confidence=0.0,
                        score=detection.confidence * 0.4,
                        valid=False,
                        plate_color=plate_color,
                        rectified_size=(
                            int(plate_crop.shape[1]),
                            int(plate_crop.shape[0]),
                        ),
                        processing_time_ms=(time.perf_counter() - start) * 1000.0,
                        candidates=[],
                        error="OCR returned no usable text.",
                        accepted=False,
                        status="no_text",
                    )
                )
                continue

            best = formatted[0]
            accepted = (
                best.valid
                and best.confidence >= self.config.ocr.accept_confidence
            )
            status = "accepted" if accepted else "rejected"
            error = None
            if not accepted:
                error = (
                    "low_confidence"
                    if best.confidence < self.config.ocr.accept_confidence
                    else "invalid_plate_format"
                )
            score = (
                0.35 * detection.confidence
                + 0.45 * best.confidence
                + 0.20 * best.format_score
                + (0.05 if best.valid else 0.0)
            )
            if not quad_found:
                score *= 0.97
            results.append(
                PlateResult(
                    image_path=source,
                    bbox=detection.bbox,
                    detection_confidence=round(detection.confidence, 4),
                    plate_text=best.text if accepted else "",
                    ocr_confidence=round(best.confidence, 4),
                    score=round(min(1.0, score), 4),
                    valid=best.valid,
                    plate_color=plate_color,
                    rectified_size=(
                        int(plate_crop.shape[1]),
                        int(plate_crop.shape[0]),
                    ),
                    processing_time_ms=round(
                        (time.perf_counter() - start) * 1000.0, 2
                    ),
                    candidates=formatted[:6],
                    accepted=accepted,
                    status=status,
                    error=error,
                )
            )

        results.sort(key=lambda item: (item.valid, item.score), reverse=True)
        deduplicated: list[PlateResult] = []
        for result in results:
            if any(
                _box_iou(result.bbox, kept.bbox) > 0.60
                and result.plate_text == kept.plate_text
                for kept in deduplicated
            ):
                continue
            deduplicated.append(result)
        return deduplicated

    def process_image(self, image_path: str | Path) -> list[PlateResult]:
        path = Path(image_path).resolve()
        image = read_image(path)
        return self.process_array(image, source=str(path))