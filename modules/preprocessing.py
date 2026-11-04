"""Image I/O and OCR-oriented preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read a BGR image, including paths containing Chinese characters."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image without relying on OpenCV Unicode path support."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode image for {output}")
    encoded.tofile(str(output))


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def clahe_bgr(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    bgr = ensure_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(8, 8)
    ).apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def gamma_correct(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    inv_gamma = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(np.uint8)
    return cv2.LUT(ensure_bgr(image), table)


def make_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate a small set of complementary inputs for PaddleOCR.

    We intentionally avoid binarization by default: general-purpose OCR models
    usually retain more information from a color or contrast-enhanced crop.
    """
    color = ensure_bgr(image)
    variants = [("color", color), ("clahe", clahe_bgr(color))]
    if float(color.mean()) < 85.0:
        variants.append(("gamma", gamma_correct(color, gamma=1.35)))
    return variants


def resize_for_display(image: np.ndarray, max_side: int = 1400) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )