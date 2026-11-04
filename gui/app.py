#!/usr/bin/env python3
"""Tkinter desktop demo for the YOLO + PaddleOCR pipeline."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from config import PATHS, ensure_directories
from modules.pipeline import LicensePlatePipeline
from modules.preprocessing import read_image, resize_for_display
from utils.visualization import draw_results


class LicensePlateApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLOv8n + PaddleOCR 车牌识别")
        self.root.geometry("1180x760")
        self.root.configure(bg="#f3f5f8")
        self.pipeline: LicensePlatePipeline | None = None
        self.image_path: Path | None = None
        self.original = None
        self.preview: ImageTk.PhotoImage | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        header = ttk.Frame(root, padding=(22, 18))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="YOLOv8n + PaddleOCR",
            font=("Microsoft YaHei UI", 22, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="目标检测负责定位车牌，PaddleOCR 识别分支负责读取字符。",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        actions = ttk.Frame(root, padding=(22, 0))
        actions.pack(fill="x", pady=(0, 12))
        ttk.Button(actions, text="选择图片", command=self.choose_image).pack(side="left")
        self.run_button = ttk.Button(actions, text="开始识别", command=self.recognize)
        self.run_button.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="等待输入图片")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=12)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=22, pady=(0, 22))

        preview_frame = ttk.Frame(body, padding=8)
        body.add(preview_frame, weight=3)
        self.image_label = ttk.Label(preview_frame, text="图片预览", anchor="center")
        self.image_label.pack(fill="both", expand=True)

        result_frame = ttk.Frame(body, padding=14)
        body.add(result_frame, weight=2)
        ttk.Label(
            result_frame,
            text="识别结果",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).pack(anchor="w")
        self.result_text = tk.Text(
            result_frame,
            width=38,
            height=25,
            font=("Microsoft YaHei UI", 10),
            wrap="word",
            borderwidth=0,
            relief="flat",
        )
        self.result_text.pack(fill="both", expand=True, pady=(8, 0))

        self.root.after(100, self.poll_events)

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=PATHS.test,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not selected:
            return
        self.image_path = Path(selected)
        self.original = read_image(self.image_path)
        self._show_image(self.original)
        self.status_var.set(self.image_path.name)
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "图片已载入，点击“开始识别”。\n")

    def _show_image(self, image) -> None:
        display = resize_for_display(image, max_side=780)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.preview = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=self.preview, text="")

    def recognize(self) -> None:
        if self.image_path is None:
            messagebox.showinfo("提示", "请先选择图片。")
            return
        self.run_button.configure(state="disabled")
        self.status_var.set("模型加载/识别中...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        started = time.perf_counter()
        try:
            ensure_directories()
            if self.pipeline is None:
                self.pipeline = LicensePlatePipeline()
            assert self.image_path is not None
            results = self.pipeline.process_image(self.image_path)
            elapsed = (time.perf_counter() - started) * 1000.0
            self.events.put(("done", (results, elapsed)))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "error":
                    self.status_var.set("识别失败")
                    self.run_button.configure(state="normal")
                    messagebox.showerror("识别失败", str(payload))
                elif kind == "done":
                    results, elapsed = payload  # type: ignore[misc]
                    self._render_results(results, elapsed)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def _render_results(self, results, elapsed: float) -> None:
        assert self.original is not None
        annotated = draw_results(self.original, results)
        self._show_image(annotated)
        self.run_button.configure(state="normal")
        self.status_var.set(f"完成，用时 {elapsed:.1f} ms")
        self.result_text.delete("1.0", "end")
        if not results:
            self.result_text.insert("end", "未检测到车牌。\n")
            return
        for index, result in enumerate(results, start=1):
            self.result_text.insert(
                "end",
                f"候选 {index}\n"
                f"车牌号：{result.plate_text or '未识别'}\n"
                f"颜色：{result.plate_color}\n"
                f"格式有效：{'是' if result.valid else '否'}\n"
                f"检测置信度：{result.detection_confidence:.3f}\n"
                f"OCR 置信度：{result.ocr_confidence:.3f}\n"
                f"综合评分：{result.score:.3f}\n"
                f"矫正：{result.rectified_size[0]} × {result.rectified_size[1]}\n"
                f"耗时：{result.processing_time_ms:.1f} ms\n\n",
            )


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    LicensePlateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()