from __future__ import annotations

import unittest

from utils.evaluation import (
    character_accuracy,
    character_error_rate,
    edit_distance,
    evaluate_results,
    iou_xyxy,
)


class EvaluationTests(unittest.TestCase):
    def test_iou_exact_box(self) -> None:
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_iou_disjoint_boxes(self) -> None:
        self.assertEqual(iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("京A12345", "京A12346"), 1)

    def test_cer_and_character_accuracy(self) -> None:
        self.assertEqual(character_error_rate("京A12345", "京A12345"), 0.0)
        self.assertEqual(character_accuracy("京A12345", "京A12345"), 1.0)

    def test_end_to_end_report(self) -> None:
        truth = {
            "a.jpg": {"plate_text": "京A12345", "box": (10, 10, 100, 50)},
            "b.jpg": {"plate_text": "冀RDC3750", "box": None},
        }
        predictions = [
            {
                "image_path": "a.jpg",
                "plate_text": "京A12345",
                "bbox": [10, 10, 100, 50],
                "score": 0.95,
                "processing_time_ms": 20,
            },
            {
                "image_path": "b.jpg",
                "plate_text": "冀RDC3751",
                "bbox": [0, 0, 100, 50],
                "score": 0.70,
                "processing_time_ms": 30,
            },
        ]
        report = evaluate_results(predictions, truth)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["localized"], 2)
        self.assertEqual(report["overall_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()