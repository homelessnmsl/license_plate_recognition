"""PaddleOCR recognition-only adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import OCR, OCRConfig
from modules.preprocessing import make_ocr_variants
from modules.schemas import OCRCandidate


class OCRUnavailableError(RuntimeError):
    """Raised when PaddleOCR or its model files cannot be initialized."""


class PaddlePlateRecognizer:
    """Run PaddleOCR's recognition branch on a plate crop.

    PaddleOCR 2.x is pinned in requirements because its ``ocr(..., det=False)``
    API provides a stable recognition-only mode. Detection is handled by YOLO.
    """

    def __init__(self, config: OCRConfig = OCR) -> None:
        self.config = config
        self._ocr: Any | None = None

    def _load(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRUnavailableError(
                "PaddleOCR is not installed. Install requirements first."
            ) from exc

        try:
            self._ocr = PaddleOCR(
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.lang,
                use_gpu=self.config.use_gpu,
                show_log=False,
            )
        except Exception as exc:
            raise OCRUnavailableError(
                "PaddleOCR initialization failed. Check the PaddlePaddle wheel "
                "and first-run model download."
            ) from exc

    @staticmethod
    def _walk_results(node: Any) -> list[tuple[str, float]]:
        """Extract (text, score) pairs from PaddleOCR's nested return value."""
        readings: list[tuple[str, float]] = []
        if isinstance(node, (list, tuple)):
            if (
                len(node) == 2
                and isinstance(node[0], str)
                and isinstance(node[1], (int, float))
            ):
                return [(node[0], float(node[1]))]
            for item in node:
                readings.extend(PaddlePlateRecognizer._walk_results(item))
        return readings

    def _candidates_from_result(
        self, result: Any, variant_name: str
    ) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        for text, confidence in self._walk_results(result):
            text = str(text).strip()
            if not text or confidence < self.config.min_score:
                continue
            candidates.append(
                OCRCandidate(
                    text=text,
                    confidence=float(confidence),
                    raw_text=text,
                    variant=variant_name,
                )
            )
        return candidates

    def recognize(self, plate_image: np.ndarray) -> list[OCRCandidate]:
        self._load()
        assert self._ocr is not None
        candidates: list[OCRCandidate] = []
        seen: set[str] = set()

        for variant_name, variant in make_ocr_variants(plate_image):
            try:
                result = self._ocr.ocr(
                    variant, det=False, rec=True, cls=self.config.use_angle_cls
                )
            except TypeError as exc:
                raise OCRUnavailableError(
                    "PaddleOCR API mismatch. This project pins paddleocr==2.9.1; "
                    "reinstall the versions from requirements.txt."
                ) from exc
            except Exception as exc:
                raise OCRUnavailableError(f"PaddleOCR inference failed: {exc}") from exc

            for candidate in self._candidates_from_result(result, variant_name):
                key = candidate.text.upper()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def recognize_full(self, plate_image: np.ndarray) -> list[OCRCandidate]:
        """Fallback using PaddleOCR text detection plus recognition.

        Recognition-only mode is faster and usually better for tight plate
        crops, but detection+recognition can recover characters when the crop
        contains background or when the recognition model drops a character.
        """
        self._load()
        assert self._ocr is not None
        try:
            result = self._ocr.ocr(
                plate_image, det=True, rec=True, cls=self.config.use_angle_cls
            )
        except TypeError as exc:
            raise OCRUnavailableError(
                "PaddleOCR API mismatch while running full OCR fallback."
            ) from exc
        except Exception as exc:
            raise OCRUnavailableError(f"PaddleOCR full OCR failed: {exc}") from exc
        return self._candidates_from_result(result, "full_det")