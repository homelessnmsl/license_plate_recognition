#!/usr/bin/env python3
"""Launch PP-OCRv4 recognition fine-tuning with the GPU Paddle runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_TRAIN_ROOT = Path(r"E:\license_plate_data\PaddleOCR_train")
DEFAULT_GPU_PYTHON = Path(r"E:\venvs\paddle_gpu311\Scripts\python.exe")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune PP-OCRv4 plate recognizer.")
    parser.add_argument("--config", type=Path, default=Path(r"E:\license_plate_data\ocr_finetune.yml"))
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_GPU_PYTHON)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train_script = args.training_root / "tools" / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"PaddleOCR training code not found: {train_script}")
    if not args.python.is_file():
        raise FileNotFoundError(f"GPU Python not found: {args.python}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    nvidia = args.python.parent.parent / "Lib" / "site-packages" / "nvidia"
    for bin_dir in (nvidia / "cudnn" / "bin", nvidia / "cublas" / "bin", nvidia / "cuda_nvrtc" / "bin"):
        if bin_dir.is_dir():
            env["PATH"] = f"{bin_dir};{env.get('PATH', '')}"

    overrides = [
        f"Global.epoch_num={args.epochs}",
        f"Train.loader.batch_size_per_card={args.batch}",
        f"Eval.loader.batch_size_per_card={args.batch}",
        "Global.use_gpu=False" if args.cpu else "Global.use_gpu=True",
    ]
    command = [str(args.python), str(train_script), "-c", str(args.config), "-o", *overrides]
    print("Running:", " ".join(command))
    return subprocess.run(command, cwd=args.training_root, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())