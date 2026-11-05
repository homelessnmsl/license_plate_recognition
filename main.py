#!/usr/bin/env python3
"""Command-line entry point for the YOLOv8n + PaddleOCR system."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from config import CONFIG, PATHS, ensure_directories
from utils.file_utils import iter_images, save_csv, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="License plate recognition: YOLOv8n detection + PaddleOCR."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", "-i", type=Path, help="Image file or directory.")
    source.add_argument("--video", "-v", type=Path, help="Video file.")
    source.add_argument("--camera", type=int, help="Camera device index.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=CONFIG.detector.weights,
        help="Trained YOLO plate weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.output,
        help="Output root directory.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CUDA device 0 for YOLO.")
    parser.add_argument(
        "--ocr-gpu",
        action="store_true",
        help="Use GPU for PaddleOCR when the installed Paddle build supports it.",
    )
    parser.add_argument("--show", action="store_true", help="Show image/video windows.")
    parser.add_argument("--frame-skip", type=int, default=CONFIG.frame_skip)
    parser.add_argument("--limit", type=int, default=0, help="Limit batch images.")
    return parser


def _make_pipeline(args: argparse.Namespace):
    from modules.pipeline import LicensePlatePipeline

    detector_config = replace(
        CONFIG.detector,
        weights=args.weights.resolve(),
        device="0" if args.gpu else CONFIG.detector.device,
    )
    ocr_config = replace(CONFIG.ocr, use_gpu=args.ocr_gpu)
    pipeline_config = replace(
        CONFIG,
        detector=detector_config,
        ocr=ocr_config,
        frame_skip=args.frame_skip,
    )
    return LicensePlatePipeline(config=pipeline_config)


def _print_results(results: list) -> None:
    if not results:
        print("未检测到车牌。")
        return
    for index, result in enumerate(results, start=1):
        print(
            f"[{index}] {result.plate_text or '<empty>'} | "
            f"color={result.plate_color} | status={result.status} | valid={result.valid} | "
            f"det={result.detection_confidence:.3f} | "
            f"ocr={result.ocr_confidence:.3f} | score={result.score:.3f} | "
            f"time={result.processing_time_ms:.1f} ms"
        )


def _run_images(pipeline, input_path: Path, output_root: Path, show: bool, limit: int) -> int:
    images = iter_images(input_path)
    if limit > 0:
        images = images[:limit]
    if not images:
        raise ValueError(f"No supported images found: {input_path}")

    json_dir = output_root / "json"
    image_dir = output_root / "images"
    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}")
        started = time.perf_counter()
        results = pipeline.process_image(image_path)
        if not results:
            rows.append(
                {
                    "image_path": str(image_path),
                    "plate_text": "",
                    "valid": False,
                    "score": 0.0,
                    "error": "no_plate_detected",
                }
            )
        else:
            for result in results:
                row = result.to_dict()
                row["processing_time_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 2
                )
                rows.append(row)
        _print_results(results)

        import cv2

        from modules.preprocessing import read_image, write_image
        from utils.visualization import draw_results

        original = read_image(image_path)
        annotated = draw_results(original, results)
        write_image(image_dir / f"{image_path.stem}_result.jpg", annotated)
        if show:
            cv2.imshow("YOLO + PaddleOCR", annotated)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    save_json(json_dir / "results.json", rows)
    save_csv(
        json_dir / "results.csv",
        rows,
        [
            "image_path",
            "plate_text",
            "valid",
            "plate_color",
            "detection_confidence",
            "ocr_confidence",
            "score",
            "bbox",
            "processing_time_ms",
            "status",
            "accepted",
            "error",
        ],
    )
    print(f"结果已保存到: {output_root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_directories()
    try:
        pipeline = _make_pipeline(args)
        if args.image:
            return _run_images(pipeline, args.image, args.output, args.show, args.limit)
        if args.video:
            if not args.video.is_file():
                raise FileNotFoundError(args.video)
            from modules.video import process_video

            output_dir = args.output / "video"
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = process_video(
                pipeline,
                args.video,
                output_dir / f"{args.video.stem}_result.mp4",
                frame_skip=args.frame_skip,
                show=args.show,
            )
            save_json(output_dir / f"{args.video.stem}_summary.json", summary)
            print(summary)
            return 0
        if args.camera is not None:
            from modules.video import process_video

            summary = process_video(
                pipeline,
                args.camera,
                frame_skip=args.frame_skip,
                show=True,
            )
            print(summary)
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())