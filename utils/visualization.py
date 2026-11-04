"""Chinese-capable result visualization."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.schemas import PlateResult


def _font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_results(image: np.ndarray, results: list[PlateResult]) -> np.ndarray:
    """Draw YOLO boxes and OCR text without OpenCV Chinese glyph issues."""
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = _font(22)
    small = _font(17)

    for index, result in enumerate(results, start=1):
        x1, y1, x2, y2 = result.bbox
        color = (35, 190, 90) if result.valid else (240, 160, 30)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = (
            f"{index}. {result.plate_text or '未识别'}  "
            f"{result.score:.2f}  {result.plate_color}"
        )
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_height = bottom - top + 10
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle(
            (x1, label_y, min(canvas.width, x1 + right - left + 14), label_y + text_height),
            fill=color,
        )
        draw.text((x1 + 6, label_y + 4), label, fill=(255, 255, 255), font=font)

    header = f"YOLOv8n + PaddleOCR | plates: {len(results)}"
    draw.rectangle((0, 0, min(canvas.width, 460), 30), fill=(20, 30, 45))
    draw.text((8, 5), header, fill=(255, 255, 255), font=small)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)