"""Detection and OCR evaluation metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def character_accuracy(reference: str, hypothesis: str) -> float:
    return max(0.0, 1.0 - character_error_rate(reference, hypothesis))


def load_ground_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load CSV with image_path,plate_text and optional x1,y1,x2,y2."""
    truth: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = (row.get("image_path") or "").strip().replace("\\", "/")
            text = (row.get("plate_text") or "").strip()
            if not image_path or not text:
                continue
            box = None
            if all(row.get(key) not in (None, "") for key in ("x1", "y1", "x2", "y2")):
                box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            truth[image_path] = {"plate_text": text, "box": box}
    return truth


def evaluate_results(
    predictions: Iterable[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    grouped_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        key = str(row.get("image_path", "")).replace("\\", "/")
        grouped.setdefault(key, []).append(row)
        grouped_by_name.setdefault(Path(key).name, []).append(row)

    total = localized = recognized = 0
    char_correct = total_chars = 0
    times: list[float] = []
    details: list[dict[str, Any]] = []

    for path, item in truth.items():
        total += 1
        reference = item["plate_text"]
        normalized_path = path.replace("\\", "/")
        candidates = grouped.get(normalized_path, [])
        if not candidates:
            candidates = grouped_by_name.get(Path(normalized_path).name, [])
        best = max(candidates, key=lambda row: float(row.get("score", 0.0)), default={})

        pred_box = best.get("bbox")
        localized_ok = bool(pred_box)
        if localized_ok and item.get("box"):
            localized_ok = iou_xyxy(tuple(pred_box), tuple(item["box"])) >= iou_threshold
        if localized_ok:
            localized += 1

        hypothesis = str(best.get("plate_text", ""))
        exact = localized_ok and hypothesis == reference
        if localized_ok and hypothesis:
            recognized += 1
        total_chars += len(reference)
        char_correct += max(0, len(reference) - edit_distance(reference, hypothesis))
        if "processing_time_ms" in best:
            times.append(float(best["processing_time_ms"]))
        details.append(
            {
                "image": path,
                "reference": reference,
                "hypothesis": hypothesis,
                "localized": localized_ok,
                "exact_match": exact,
                "cer": round(character_error_rate(reference, hypothesis), 4),
            }
        )

    n = float(total) if total else 1.0
    return {
        "total": total,
        "localized": localized,
        "recognized": recognized,
        "localization_accuracy": round(localized / n, 4),
        "recognition_accuracy": round(recognized / localized, 4) if localized else 0.0,
        "overall_accuracy": round(
            sum(item["exact_match"] for item in details) / n, 4
        ),
        "character_accuracy": round(char_correct / total_chars, 4) if total_chars else 0.0,
        "mean_processing_time_ms": round(sum(times) / len(times), 2) if times else 0.0,
        "details": details,
    }