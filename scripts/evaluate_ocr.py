#!/usr/bin/env python3
"""Evaluate the fine-tuned PaddleOCR recognizer on plate crops."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.postprocess import make_candidate, rank_candidates
from modules.preprocessing import read_image
from utils.evaluation import character_error_rate
from utils.file_utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy on crops.")
    parser.add_argument("--labels", type=Path, default=Path(r"E:\license_plate_data\ocr\labels\test.txt"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\license_plate_data\ocr"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "json" / "ocr_evaluation.json")
    args = parser.parse_args()

    if args.model_dir:
        os.environ["LPR_OCR_REC_MODEL_DIR"] = str(args.model_dir)
    from modules.recognizer import PaddlePlateRecognizer

    recognizer = PaddlePlateRecognizer()
    details = []
    exact = 0
    char_acc = 0.0
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        relative, truth = line.split("\t", 1)
        image = read_image(args.data_root / relative)
        candidates = rank_candidates(
            [make_candidate(item.text, item.confidence, item.variant) for item in recognizer.recognize(image)]
        )
        predicted = candidates[0].text if candidates else ""
        match = predicted == truth
        cer = character_error_rate(truth, predicted)
        exact += int(match)
        char_acc += 1.0 - cer
        details.append({"image": relative, "truth": truth, "predicted": predicted, "exact_match": match, "cer": round(cer, 4)})

    report = {
        "total": len(details),
        "exact_matches": exact,
        "exact_accuracy": round(exact / len(details), 4) if details else 0.0,
        "char_accuracy": round(char_acc / len(details), 4) if details else 0.0,
        "details": details,
    }
    save_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())