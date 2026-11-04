#!/usr/bin/env python3
"""Evaluate saved predictions against a labeled test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS
from utils.evaluation import evaluate_results, load_ground_truth
from utils.file_utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end plate recognition.")
    parser.add_argument(
        "--predictions", type=Path, default=PATHS.json / "results.json"
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=PATHS.test / "ground_truth.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PATHS.json / "evaluation.json"
    )
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    truth = load_ground_truth(args.ground_truth)
    report = evaluate_results(predictions, truth, iou_threshold=args.iou)
    save_json(args.output, report)
    summary = {key: value for key, value in report.items() if key != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())