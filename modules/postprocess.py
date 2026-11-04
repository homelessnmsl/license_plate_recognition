"""Chinese license-plate normalization, validation and candidate ranking."""

from __future__ import annotations

import re
import unicodedata

from modules.schemas import OCRCandidate


PROVINCES = frozenset("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")
SPECIAL_PREFIXES = frozenset("使领警学挂港澳")
VALID_CHARS = frozenset("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
_CJK_OR_ALNUM = re.compile(r"[^\u4e00-\u9fffA-Z0-9]")


def normalize_text(text: str) -> str:
    """Normalize width/case and remove separators or OCR punctuation."""
    normalized = unicodedata.normalize("NFKC", text or "").upper()
    normalized = normalized.replace("·", "").replace("•", "")
    return _CJK_OR_ALNUM.sub("", normalized)


def trim_to_province(text: str) -> str:
    normalized = normalize_text(text)
    for index, char in enumerate(normalized):
        if char in PROVINCES or char in SPECIAL_PREFIXES:
            return normalized[index:]
    return normalized


def apply_position_corrections(text: str) -> str:
    """Apply only safe plate-position corrections.

    I and O are not used in the serial portion of ordinary Chinese plates, so
    OCR confusions in positions 3 onwards can be mapped to 1 and 0. The second
    character is preserved because it is normally a letter.
    """
    if len(text) <= 2:
        return text
    prefix = text[:2]
    tail = text[2:].replace("I", "1").replace("O", "0")
    return prefix + tail


def format_score(text: str) -> float:
    score = 0.0
    if len(text) in {7, 8}:
        score += 0.35
    if text and (text[0] in PROVINCES or text[0] in SPECIAL_PREFIXES):
        score += 0.30
    if len(text) >= 2 and text[1] in VALID_CHARS and text[1].isalpha():
        score += 0.20
    if text and all(char in VALID_CHARS for char in text[1:]):
        score += 0.15
    return min(1.0, score)


def is_valid_plate(text: str) -> bool:
    if len(text) not in {7, 8}:
        return False
    if text[0] not in PROVINCES and text[0] not in SPECIAL_PREFIXES:
        return False
    if len(text) >= 2 and (not text[1].isalpha() or text[1] not in VALID_CHARS):
        return False
    return all(char in VALID_CHARS for char in text[1:])


def clean_candidate(text: str) -> str:
    trimmed = trim_to_province(text)
    return apply_position_corrections(trimmed)


def make_candidate(
    raw_text: str,
    confidence: float,
    variant: str = "color",
) -> OCRCandidate:
    cleaned = clean_candidate(raw_text)
    score = format_score(cleaned)
    return OCRCandidate(
        text=cleaned,
        confidence=max(0.0, min(1.0, float(confidence))),
        raw_text=raw_text,
        valid=is_valid_plate(cleaned),
        format_score=score,
        variant=variant,
    )


def rank_candidates(candidates: list[OCRCandidate]) -> list[OCRCandidate]:
    return sorted(
        candidates,
        key=lambda item: (item.valid, item.confidence, item.format_score, len(item.text)),
        reverse=True,
    )