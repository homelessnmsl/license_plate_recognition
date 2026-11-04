from __future__ import annotations

import unittest

from scripts.prepare_ccpd import parse_ccpd_stem


class CCPDParserTests(unittest.TestCase):
    def test_comma_separated_bbox(self) -> None:
        stem = (
            "0189655172414-91_85-233,591_457,681-"
            "463,678_245,663_232,585_450,600-"
            "0_3_15_27_28_29_30-134-129_ccpd_base_005128"
        )
        parsed = parse_ccpd_stem(stem)
        self.assertIsNotNone(parsed)
        plate, box = parsed
        self.assertEqual(box, (233, 591, 457, 681))
        self.assertEqual(plate[0], "皖")
        self.assertEqual(len(plate), 7)

    def test_ampersand_separated_bbox(self) -> None:
        stem = (
            "025-95_113-154&383_386&473-"
            "386&473_177&473_154&383_154&383-"
            "0_0_22_27_27_33_16-37-15"
        )
        parsed = parse_ccpd_stem(stem)
        self.assertIsNotNone(parsed)
        _, box = parsed
        self.assertEqual(box, (154, 383, 386, 473))

    def test_test_set_prefix(self) -> None:
        stem = (
            "test_0525_0087-4_3-247,535_359,600-"
            "358,600_247,591_248,535_359,544-"
            "0_0_20_3_27_27_33-104-8_ccpd_challenge_058809"
        )
        parsed = parse_ccpd_stem(stem)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[1], (247, 535, 359, 600))


if __name__ == "__main__":
    unittest.main()