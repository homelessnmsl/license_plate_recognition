# 环境配置

## 推荐环境

| 项目 | 建议 |
|---|---|
| 操作系统 | Windows 10/11 |
| Python | 3.11 |
| GPU | NVIDIA RTX 3050 4GB 或更高 |
| YOLO | Ultralytics + PyTorch CUDA |
| OCR | PaddlePaddle 2.6.2 + PaddleOCR 2.9.1 |
| OCR 设备 | 默认 CPU，避免 Windows 下 Paddle GPU 兼容问题 |

当前 C 盘剩余空间较少，建议虚拟环境放在 D 盘或 E 盘，例如：

```powershell
D:\venvs\lpr311\Scripts\Activate.ps1
```

## 安装顺序

### 1. 创建 Python 3.11 环境

```powershell
py -3.11 -m venv D:\venvs\lpr311
D:\venvs\lpr311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 CUDA 版 PyTorch

如果使用 NVIDIA GPU：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

如果只使用 CPU：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装项目依赖

进入项目目录：

```powershell
cd C:\Users\HP\Desktop\license_plate_recognition
pip install -r requirements.txt
```

PaddleOCR 第一次运行会自动下载中文检测/识别模型。模型下载需要网络。

### 4. 检查环境

```powershell
python scripts/check_environment.py
```

期望确认：

- `torch.cuda.is_available() == True`
- `ultralytics` 可导入
- `paddleocr` 可导入
- OpenCV 可导入

## 运行参数

默认 YOLO 权重：

```text
data/models/yolov8n_plate.pt
```

默认输出：

```text
outputs/json
outputs/images
outputs/video
```

常用环境变量：

```powershell
$env:LPR_DEVICE = "0"
$env:LPR_OCR_USE_GPU = "false"
```

## Windows 注意事项

1. PowerShell 首次执行脚本可能需要调整执行策略，按学校环境要求处理。
2. PaddleOCR 和 OpenCV 都依赖 NumPy，必须使用 `numpy<2.0`，否则 PaddleOCR 2.9.1 可能出现兼容问题。
3. 不要同时运行两个大型 GPU 推理任务，4GB 显存容易不足。
4. 初次模型下载或 OCR 初始化可能较慢，不代表程序卡死。