"""File and export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from config import CONFIG


def iter_images(root: str | Path, recursive: bool = True) -> list[Path]:
    path = Path(root)
    extensions = CONFIG.image_extensions
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    if not path.exists():
        return []
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        item
        for item in iterator
        if item.is_file() and item.suffix.lower() in extensions
    )


def save_json(path: str | Path, data: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def save_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output