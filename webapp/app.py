"""FastAPI deployment service for the license-plate recognition pipeline."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CONFIG, PATHS
from modules.pipeline import LicensePlatePipeline
from modules.preprocessing import resize_for_display
from utils.visualization import draw_results


LOGGER = logging.getLogger("lpr.web")
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "webapp" / "static"
DEMO_DIR = ROOT / "data" / "demo"
DEPLOY_MODELS = ROOT / "deploy" / "models"


@dataclass(frozen=True)
class WebSettings:
    access_key: str = os.getenv("LPR_WEB_TOKEN", "").strip()
    max_upload_mb: int = int(os.getenv("LPR_MAX_UPLOAD_MB", "15"))
    max_image_side: int = int(os.getenv("LPR_MAX_IMAGE_SIDE", "2200"))
    max_pixels: int = int(os.getenv("LPR_MAX_PIXELS", "40000000"))
    rate_limit_per_minute: int = int(os.getenv("LPR_RATE_LIMIT", "30"))
    warmup: bool = os.getenv("LPR_WARMUP", "true").lower() not in {"0", "false", "no"}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@dataclass
class Runtime:
    pipeline: LicensePlatePipeline | None = None
    ready: bool = False
    warmup_ms: float = 0.0
    error: str = ""


SETTINGS = WebSettings()
RUNTIME = Runtime()
INFERENCE_LOCK = threading.Lock()


class RateLimiter:
    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = max(1, limit)
        self.window_seconds = window_seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            bucket = self.events[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


RATE_LIMITER = RateLimiter(SETTINGS.rate_limit_per_minute)


def build_pipeline() -> LicensePlatePipeline:
    yolo_env = os.getenv("LPR_YOLO_WEIGHTS", "").strip()
    onnx_model = DEPLOY_MODELS / "yolov8n_plate.onnx"
    weights = Path(yolo_env) if yolo_env else (
        onnx_model if onnx_model.is_file() else CONFIG.detector.weights
    )

    ocr_env = os.getenv("LPR_OCR_REC_MODEL_DIR", "").strip()
    deployed_ocr = DEPLOY_MODELS / "ocr_rec"
    local_ocr = PATHS.models / "ocr_rec"
    if ocr_env:
        ocr_dir = Path(ocr_env)
    elif (deployed_ocr / "inference.pdmodel").is_file():
        ocr_dir = deployed_ocr
    elif (local_ocr / "inference.pdmodel").is_file():
        ocr_dir = local_ocr
    else:
        ocr_dir = CONFIG.ocr.rec_model_dir

    detector = replace(
        CONFIG.detector,
        weights=weights.resolve(),
        device=os.getenv("LPR_DEVICE", "").strip() or None,
    )
    ocr = replace(CONFIG.ocr, rec_model_dir=ocr_dir)
    pipeline_config = replace(CONFIG, detector=detector, ocr=ocr)
    return LicensePlatePipeline(config=pipeline_config)


def warmup_pipeline(pipeline: LicensePlatePipeline) -> None:
    samples = sorted(DEMO_DIR.glob("*.jpg"))
    if not samples:
        # Still force both model loaders even without a demo image.
        pipeline.detector._load()  # type: ignore[attr-defined]
        pipeline.recognizer._load()  # type: ignore[attr-defined]
        return
    image = cv2.imdecode(np.fromfile(str(samples[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        pipeline.process_array(image, source="<warmup>")


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    try:
        pipeline = await asyncio.to_thread(build_pipeline)
        if SETTINGS.warmup:
            await asyncio.to_thread(warmup_pipeline, pipeline)
        RUNTIME.pipeline = pipeline
        RUNTIME.ready = True
        RUNTIME.warmup_ms = round((time.perf_counter() - started) * 1000.0, 1)
        LOGGER.info("Model ready in %.1f ms", RUNTIME.warmup_ms)
    except Exception as exc:  # Keep the health endpoint available for diagnosis.
        RUNTIME.error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Model initialization failed")
    yield
    RUNTIME.ready = False


app = FastAPI(
    title="车牌识别 Web 服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def request_key(request: Request) -> str:
    return request.headers.get("X-Access-Key", "").strip() or request.query_params.get(
        "key", ""
    ).strip()


def require_access(request: Request) -> None:
    if not SETTINGS.access_key:
        return
    supplied = request_key(request)
    if not supplied or not secrets.compare_digest(supplied, SETTINGS.access_key):
        raise HTTPException(status_code=401, detail="访问码无效或缺失")


def client_ip(request: Request) -> str:
    return (
        request.headers.get("CF-Connecting-IP", "").strip()
        or (request.client.host if request.client else "unknown")
    )


def encode_jpeg(image: np.ndarray, max_side: int = 1600, quality: int = 88) -> str:
    display = resize_for_display(image, max_side=max_side)
    ok, encoded = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode annotated image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    prefix = "data:" + "image/jpeg" + ";base64,"
    return prefix + payload


def serialize_plate(result: Any) -> dict[str, Any]:
    best_text = result.candidates[0].text if result.candidates else ""
    return {
        "text": result.plate_text,
        "best_text": best_text,
        "accepted": bool(result.accepted),
        "status": result.status,
        "valid": bool(result.valid),
        "plate_color": result.plate_color,
        "score": round(float(result.score), 4),
        "detection_confidence": round(float(result.detection_confidence), 4),
        "ocr_confidence": round(float(result.ocr_confidence), 4),
        "bbox": list(result.bbox),
        "processing_time_ms": round(float(result.processing_time_ms), 1),
        "error": result.error,
    }


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.svg", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg")


@app.get("/api/info")
async def info() -> dict[str, Any]:
    return {
        "ready": RUNTIME.ready,
        "error": RUNTIME.error if not RUNTIME.ready else "",
        "auth_required": bool(SETTINGS.access_key),
        "max_upload_mb": SETTINGS.max_upload_mb,
        "warmup_ms": RUNTIME.warmup_ms,
        "models": {
            "detector": "YOLOv8n ONNX",
            "recognizer": "PP-OCRv4 中文车牌微调",
        },
    }


@app.get("/api/health")
async def health() -> JSONResponse:
    body = {"status": "ok" if RUNTIME.ready else "loading", "ready": RUNTIME.ready}
    return JSONResponse(body, status_code=200 if RUNTIME.ready else 503)


@app.post("/api/recognize", dependencies=[Depends(require_access)])
async def recognize(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if not RUNTIME.ready or RUNTIME.pipeline is None:
        raise HTTPException(status_code=503, detail="模型正在加载，请稍后重试")
    if not RATE_LIMITER.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传图片文件")
    payload = await file.read(SETTINGS.max_upload_bytes + 1)
    if len(payload) > SETTINGS.max_upload_bytes:
        raise HTTPException(
            status_code=413, detail=f"图片不能超过 {SETTINGS.max_upload_mb} MB"
        )
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解码图片，请使用 JPG/PNG/WebP")
    height, width = image.shape[:2]
    if width * height > SETTINGS.max_pixels:
        raise HTTPException(status_code=413, detail="图片像素过大，请缩小后重试")

    resized = False
    longest = max(height, width)
    if longest > SETTINGS.max_image_side:
        scale = SETTINGS.max_image_side / float(longest)
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        resized = True

    started = time.perf_counter()
    try:
        with INFERENCE_LOCK:
            results = await asyncio.to_thread(
                RUNTIME.pipeline.process_array, image, "<upload>"
            )
        annotated = await asyncio.to_thread(draw_results, image, results)
        annotated_data = await asyncio.to_thread(encode_jpeg, annotated)
    except Exception as exc:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"识别失败: {exc}") from exc

    return {
        "ok": True,
        "plates": [serialize_plate(item) for item in results],
        "accepted_count": sum(1 for item in results if item.accepted),
        "annotated_image": annotated_data,
        "image": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "resized": resized,
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if DEMO_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR), name="demo")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "webapp.app:app",
        host=os.getenv("LPR_HOST", "0.0.0.0"),
        port=int(os.getenv("LPR_PORT", "8000")),
        workers=1,
    )

