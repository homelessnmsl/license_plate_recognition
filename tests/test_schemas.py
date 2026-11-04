from __future__ import annotations

import unittest

from modules.schemas import Detection, OCRCandidate, PlateResult


class SchemaTests(unittest.TestCase):
    def test_detection_geometry(self) -> None:
        detection = Detection((10, 20, 110, 60), 0.9)
        self.assertEqual(detection.width, 100)
        self.assertEqual(detection.height, 40)
        self.assertEqual(detection.area, 4000)
        self.assertEqual(detection.xywh(), (10, 20, 100, 40))

    def test_plate_result_serialization(self) -> None:
        result = PlateResult(
            image_path="a.jpg",
            bbox=(1, 2, 30, 10),
            detection_confidence=0.9,
            plate_text="京A12345",
            ocr_confidence=0.8,
            score=0.85,
            valid=True,
            plate_color="blue",
            rectified_size=(300, 64),
            processing_time_ms=25.0,
            candidates=[OCRCandidate("京A12345", 0.8)],
        )
        payload = result.to_dict()
        self.assertEqual(payload["bbox"], [1, 2, 30, 10])
        self.assertEqual(payload["rectified_size"], [300, 64])
        self.assertEqual(payload["candidates"][0]["text"], "京A12345")


if __name__ == "__main__":
    unittest.main()