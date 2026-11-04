"""Frame-sampled video inference with cached detections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from config import CONFIG
from modules.pipeline import LicensePlatePipeline
from modules.schemas import PlateResult
from utils.visualization import draw_results


def process_video(
    pipeline: LicensePlatePipeline,
    source: str | Path | int,
    output_path: str | Path | None = None,
    frame_skip: int | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Process video/camera input, running full inference every N frames."""
    skip = max(1, frame_skip or CONFIG.frame_skip)
    capture_source: str | int = str(source) if isinstance(source, Path) else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Could not create output video: {output}")

    frame_index = 0
    inference_count = 0
    cached_results: list[PlateResult] = []
    all_texts: list[str] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % skip == 0:
                cached_results = pipeline.process_array(frame, source=str(source))
                inference_count += 1
                all_texts.extend(
                    item.plate_text for item in cached_results if item.plate_text
                )

            annotated = draw_results(frame, cached_results)
            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("YOLO + PaddleOCR", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    counts = Counter(all_texts)
    return {
        "source": str(source),
        "output": str(output_path) if output_path else None,
        "frames": frame_index,
        "inference_frames": inference_count,
        "frame_skip": skip,
        "plate_counts": dict(counts),
        "most_common_plate": counts.most_common(1)[0][0] if counts else None,
    }