"""Typed data structures shared by the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    class_name: str = "license_plate"

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    confidence: float
    raw_text: str = ""
    valid: bool = False
    format_score: float = 0.0
    variant: str = "color"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlateResult:
    image_path: str
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    plate_text: str
    ocr_confidence: float
    score: float
    valid: bool
    plate_color: str
    rectified_size: tuple[int, int]
    processing_time_ms: float
    candidates: list[OCRCandidate] = field(default_factory=list)
    error: str | None = None
    accepted: bool = False
    status: str = "rejected"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["rectified_size"] = list(self.rectified_size)
        data["candidates"] = [item.to_dict() for item in self.candidates]
        return data