"""Crop, color analysis and perspective rectification for plates."""

from __future__ import annotations

import cv2
import numpy as np

from modules.schemas import Detection


COLOR_RANGES = {
    "blue": ((100, 70, 40), (130, 255, 255)),
    "yellow": ((15, 65, 70), (38, 255, 255)),
    "green": ((35, 35, 35), (95, 255, 255)),
}


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width - 1)),
        max(0, min(y2, height - 1)),
    )


def crop_detection(image: np.ndarray, detection: Detection, margin: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)
    x1, y1, x2, y2 = _clamp_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height
    )
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Detected plate crop is empty")
    return crop


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    for name, (lower, upper) in COLOR_RANGES.items():
        masks[name] = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
    return masks


def classify_plate_color(image: np.ndarray) -> str:
    """Classify plate background color using HSV pixel ratios."""
    if image.size == 0:
        return "unknown"
    masks = color_masks(image)
    total = float(image.shape[0] * image.shape[1])
    ratios = {name: float(np.count_nonzero(mask)) / total for name, mask in masks.items()}
    color, ratio = max(ratios.items(), key=lambda item: item[1])
    return color if ratio >= 0.12 else "unknown"


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def find_colored_plate_quad(crop: np.ndarray) -> np.ndarray | None:
    """Find a rotated plate quadrilateral inside a YOLO crop."""
    masks = color_masks(crop)
    combined = cv2.bitwise_or(masks["blue"], masks["yellow"])
    combined = cv2.bitwise_or(combined, masks["green"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    crop_area = float(crop.shape[0] * crop.shape[1])
    best: tuple[float, np.ndarray] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < crop_area * 0.18:
            continue
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) <= 0:
            continue
        aspect = max(width, height) / min(width, height)
        if not (1.7 <= aspect <= 7.5):
            continue
        rectangularity = area / float(width * height)
        if rectangularity < 0.50:
            continue
        score = area * rectangularity
        if best is None or score > best[0]:
            best = (score, cv2.boxPoints(rect))
    return None if best is None else order_quad_points(best[1])


def warp_quad(crop: np.ndarray, quad: np.ndarray) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = order_quad_points(quad)
    width = int(
        max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    )
    height = int(
        max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    )
    if width < 2 or height < 2:
        return crop

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    warped = cv2.warpPerspective(crop, matrix, (width, height))
    if warped.shape[1] < warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def normalize_plate_size(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    scale = target_height / float(height)
    if scale > 1.0:
        return cv2.resize(
            image,
            (max(1, int(round(width * scale))), target_height),
            interpolation=cv2.INTER_CUBIC,
        )
    return image


def rectify_plate(
    image: np.ndarray,
    detection: Detection,
    margin: float = 0.08,
) -> tuple[np.ndarray, str, bool]:
    """Return (rectified_crop, plate_color, quad_was_found)."""
    crop = crop_detection(image, detection, margin=margin)
    plate_color = classify_plate_color(crop)
    quad = find_colored_plate_quad(crop)
    if quad is None:
        return normalize_plate_size(crop), plate_color, False
    rectified = normalize_plate_size(warp_quad(crop, quad))
    if rectified.size == 0:
        return normalize_plate_size(crop), plate_color, False
    rectified_color = classify_plate_color(rectified)
    if rectified_color == "unknown":
        rectified_color = plate_color
    return rectified, rectified_color, True