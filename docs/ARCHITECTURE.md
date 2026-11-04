# 系统架构

## 数据流

```text
Image / Video
   │
   ▼
preprocessing.read_image
   │
   ▼
detector.YOLOPlateDetector
   │  Detection(x1, y1, x2, y2, confidence)
   ▼
rectifier.rectify_plate
   ├─ crop_detection
   ├─ classify_plate_color
   ├─ find_colored_plate_quad
   └─ warp_quad
   │  rectified_crop, color, quad_found
   ▼
recognizer.PaddlePlateRecognizer
   │  OCRCandidate(text, confidence, variant)
   ▼
postprocess.make_candidate / rank_candidates
   │  normalized plate, valid flag, format score
   ▼
pipeline.LicensePlatePipeline
   │  PlateResult(score, bbox, color, timing)
   ▼
CLI / GUI / Video / JSON / CSV
```

## 模块边界

| 模块 | 输入 | 输出 | 责任 |
|---|---|---|---|
| `detector.py` | BGR 图像 | `Detection` | YOLO 模型加载、推理、类别过滤 |
| `rectifier.py` | 原图 + `Detection` | 矫正图、颜色 | 裁剪、颜色分析、透视矫正 |
| `recognizer.py` | 车牌裁剪图 | OCR 候选 | PaddleOCR 识别分支 |
| `postprocess.py` | 原始 OCR 文本 | 规范化候选 | 格式校验、纠错、排序 |
| `pipeline.py` | 图像路径/数组 | `PlateResult` | 流程编排、评分、去重 |
| `video.py` | 视频源 | 标注视频和统计 | 抽帧、缓存、帧写入 |
| `utils/visualization.py` | 原图 + 结果 | 标注图 | 中文显示和结果绘制 |
| `utils/evaluation.py` | 预测 + 真值 | 指标报告 | 检测、识别、字符和性能评估 |

## 依赖注入

`LicensePlatePipeline` 允许传入任意具有相同接口的检测器和识别器：

```python
pipeline = LicensePlatePipeline(detector=fake_detector, recognizer=fake_recognizer)
```

因此可以在没有 PyTorch 和 PaddlePaddle 的环境中测试流程编排、后处理和评分逻辑。

## 错误处理

- 模型权重不存在：抛出 `ModelUnavailableError`
- PaddleOCR 未安装或版本不匹配：抛出 `OCRUnavailableError`
- 图片无法读取：抛出 `FileNotFoundError` 或 `ValueError`
- 检测到框但 OCR 无文本：返回 `PlateResult(error=...)`
- 视频无法打开：抛出 `ValueError`

## 当前边界

- YOLO 权重需要训练或另行放入 `data/models/yolov8n_plate.pt`
- PaddleOCR 首次运行需要下载中文模型
- 暂时使用标准 YOLO 水平框，OBB 是后续升级方向
- 视频先使用抽帧缓存，尚未接入 ByteTrack
- OCR 直接使用通用中文模型，尚未用 CCPD 文本微调

## 升级路径

1. 标准 YOLOv8n → YOLOv8n-OBB
2. HSV 掩膜矫正 → 四角点回归
3. 抽帧缓存 → ByteTrack + 多帧投票
4. 通用 PaddleOCR → 车牌字符微调
5. OpenCV GUI → Web 演示或 FastAPI 服务