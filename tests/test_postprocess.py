from __future__ import annotations

import unittest

from modules.postprocess import (
    clean_candidate,
    format_score,
    is_valid_plate,
    make_candidate,
    normalize_text,
)


class PostprocessTests(unittest.TestCase):
    def test_removes_separator_and_spaces(self) -> None:
        self.assertEqual(normalize_text(" 京A·12345 "), "京A12345")

    def test_trims_prefix_noise(self) -> None:
        self.assertEqual(clean_candidate("NO.京A12345"), "京A12345")

    def test_valid_seven_character_plate(self) -> None:
        self.assertTrue(is_valid_plate("京A12345"))

    def test_valid_eight_character_green_plate(self) -> None:
        self.assertTrue(is_valid_plate("冀RDC3750"))

    def test_rejects_missing_province(self) -> None:
        self.assertFalse(is_valid_plate("1234567"))

    def test_position_aware_correction(self) -> None:
        self.assertEqual(clean_candidate("京AOI2345"), "京A012345")

    def test_format_score_is_bounded(self) -> None:
        self.assertGreaterEqual(format_score("京A12345"), 0.9)
        self.assertLessEqual(format_score("京A12345"), 1.0)

    def test_make_candidate_marks_valid(self) -> None:
        candidate = make_candidate("京A12345", 0.91)
        self.assertTrue(candidate.valid)
        self.assertEqual(candidate.text, "京A12345")


if __name__ == "__main__":
    unittest.main()