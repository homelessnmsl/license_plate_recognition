# 基于 YOLOv8n + PaddleOCR 的车牌识别系统

面向停车场场景的两阶段中文车牌识别项目：

```text
YOLOv8n 检测车牌 → 颜色分析与倾斜矫正 → PaddleOCR 识别车牌文本
```

本项目用于计算机视觉课程设计，重点不是简单调用两个模型，而是完整实现数据处理、检测微调、透视矫正、OCR 后处理、置信度融合、视频处理和量化评估。

## 当前状态

- 已建立模块化项目骨架
- 已实现 YOLO 检测器适配层
- 已实现车牌裁剪、HSV 颜色分析、最小外接矩形透视矫正
- 已实现 PaddleOCR 识别分支适配层
- 已实现车牌格式校验、字符纠错和候选融合
- 已实现图片、批量、视频和 GUI 入口
- 已提供 CCPD 转 YOLO 数据脚本与训练脚本
- 已提供定位率、识别率、字符准确率和耗时评估
- 已在 E 盘创建 Python 3.11 环境
- 已完成 YOLO + PaddleOCR 端到端冒烟测试
- 已准备 11,700 张 YOLO 训练/验证数据和 30 张独立测试图
- 已生成 10,554 张 PaddleOCR 微调裁剪
- 已加入低质量图片拒识策略（低置信度或非法格式不输出车牌）
- 尚未使用自训练权重完成最终测试

## 环境

虚拟环境：

```text
E:\venvs\lpr311
```

激活：

```powershell
. .\activate_env.ps1
```

当前环境包含 CUDA 12.4 版 PyTorch、Ultralytics 8.4.202、PaddleOCR 2.9.1 和 PaddlePaddle 2.6.2。完整版本见 `requirements-lock.txt`。

## 快速开始

```powershell
. .\activate_env.ps1

# 1. 下载并抽取 CCPD 子集
python scripts/prepare_ccpd_subset.py --root E:\license_plate_data --test-dir data\test

# 2. 转换为 YOLO 格式
python scripts/prepare_ccpd.py --ccpd-dir E:\license_plate_data\raw\ccpd_mixed_12k --output E:\license_plate_data\yolo --val-ratio 0.15

# 3. 训练 YOLOv8n
python scripts/train_yolo.py --epochs 50 --batch 8 --device 0

# 4. 单张图片
python main.py --image data/test/example.jpg --gpu

# 5. 批量图片
python main.py --image data/test --gpu

# 6. 视频
python main.py --video data/test/example.mp4 --gpu --frame-skip 3

# 7. GUI
python run_gui.py

# 8. 评估
python scripts/evaluate.py
```

数据目录：

```text
E:\license_plate_data\yolo
```

YOLO 默认权重路径：

```text
data/models/yolov8n_plate.pt
```

## 项目结构

```text
license_plate_recognition/
├── main.py                         # 命令行入口
├── run_gui.py                      # GUI 入口
├── activate_env.ps1                # E 盘环境激活
├── config.py                       # 路径与运行参数
├── requirements.txt
├── requirements-lock.txt
├── modules/
│   ├── schemas.py                  # 数据类型
│   ├── preprocessing.py            # 图像读取与 OCR 增强
│   ├── detector.py                 # YOLO 检测
│   ├── rectifier.py                # 颜色分析、倾斜矫正
│   ├── recognizer.py               # PaddleOCR 识别
│   ├── postprocess.py              # 车牌规则与纠错
│   ├── pipeline.py                 # 端到端流程
│   └── video.py                    # 视频抽帧推理
├── utils/
│   ├── visualization.py            # 中文结果可视化
│   ├── evaluation.py               # 指标计算
│   └── file_utils.py               # 文件导出
├── gui/app.py                      # Tkinter 界面
├── scripts/
│   ├── prepare_ccpd.py             # CCPD → YOLO
│   ├── train_yolo.py               # YOLO 微调
│   ├── evaluate.py                 # 端到端评估
│   └── check_environment.py        # 环境检查
├── configs/
├── data/
├── outputs/
├── tests/
└── docs/
```

## 核心文档

- `docs/PRINCIPLES.md`：YOLO、PaddleOCR、透视变换和置信度融合原理
- `docs/ARCHITECTURE.md`：模块职责、数据流与扩展点
- `docs/ENVIRONMENT.md`：E 盘环境、CUDA 和依赖版本
- `docs/DATASET.md`：CCPD 转换与自建测试集规范
- docs/DATA_SUMMARY.md：CCPD 子集、数据划分与基线结果
- docs/EXPERIMENTS.md：指标与消融实验设计

## 设计边界

- YOLO 只负责车牌定位，不负责字符识别
- PaddleOCR 在车牌裁剪图上只运行识别分支，不重复做文本检测
- 字符分割不是主路径，整牌序列识别更稳定
- 不使用全局 `O→0`、`I→1` 替换，纠错必须受车牌位置约束
- 视频采用抽帧推理并缓存结果，避免每帧都运行 OCR
- 测试集必须独立于训练集，不能用训练图片报告“高准确率”
- 当前 `data/models/yolov8n_plate.pt` 是冒烟测试权重，最终应替换为在 CCPD 子集上重新训练的权重