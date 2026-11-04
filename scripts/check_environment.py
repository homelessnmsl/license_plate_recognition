#!/usr/bin/env python3
"""Report runtime dependencies without importing heavy libraries."""

from __future__ import annotations

import importlib.util
import platform


def main() -> int:
    print(f"python={platform.python_version()} ({platform.architecture()[0]})")
    modules = (
        "cv2",
        "numpy",
        "torch",
        "ultralytics",
        "paddle",
        "paddleocr",
        "PIL",
        "yaml",
    )
    for module in modules:
        print(f"{module}={'installed' if importlib.util.find_spec(module) else 'missing'}")
    try:
        import torch

        print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
    except ImportError:
        print("cuda=unavailable (torch missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())