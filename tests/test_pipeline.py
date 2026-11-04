from __future__ import annotations

import importlib.util
import unittest

HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_CV2 and HAS_NUMPY:
    import numpy as np

    from modules.pipeline import LicensePlatePipeline
    from modules.schemas import Detection, OCRCandidate
else:
    np = None
    LicensePlatePipeline = None
    Detection = None
    OCRCandidate = None


class FakeDetector:
    def detect(self, image):
        return [Detection((5, 5, 150, 45), 0.95)]


class FakeRecognizer:
    def recognize(self, plate_image):
        return [OCRCandidate("京A12345", 0.96, raw_text="京A12345", variant="color")]


@unittest.skipUnless(HAS_CV2 and HAS_NUMPY, "numpy/OpenCV not installed")
class PipelineTests(unittest.TestCase):
    def test_pipeline_orchestration(self) -> None:
        image = np.full((80, 200, 3), 230, dtype=np.uint8)
        pipeline = LicensePlatePipeline(FakeDetector(), FakeRecognizer())
        results = pipeline.process_array(image, source="fake.jpg")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].plate_text, "京A12345")
        self.assertTrue(results[0].valid)


if __name__ == "__main__":
    unittest.main()