# 环境配置

## 当前已配置环境

| 项目 | 当前值 |
|---|---|
| Python | 3.11.0 |
| 虚拟环境 | `E:\venvs\lpr311` |
| PyTorch | 2.6.0 + CUDA 12.4 |
| GPU | NVIDIA RTX 3050 Laptop GPU，4GB |
| OpenCV | 4.10.0 |
| NumPy | 1.26.4 |
| Ultralytics | 8.4.202 |
| PaddlePaddle | 2.6.2 |
| PaddleOCR | 2.9.1 |
| OCR 设备 | CPU |

激活环境：

```powershell
cd C:\Users\HP\Desktop\license_plate_recognition
. .\activate_env.ps1
```

或者直接调用：

```powershell
E:\venvs\lpr311\Scripts\python.exe main.py --help
```

## 验证环境

```powershell
E:\venvs\lpr311\Scripts\python.exe scripts\check_environment.py
```

已经验证：

- `torch.cuda.is_available() == True`
- CUDA 版本为 12.4
- GPU 能被 PyTorch 识别
- PaddleOCR 2.9.1 能初始化中文模型
- YOLO 检测与 PaddleOCR 已完成端到端冒烟测试

## 已验证的样例

```text
京CHX389   blue    det=0.995  ocr=0.998  valid=True
冀RDC3750  green   det=0.989  ocr=0.970  valid=True
冀RX10J5   blue    det=0.994  ocr=0.993  valid=True
```

首次图片处理约 12 秒，其中包含 OCR 模型初始化；同一进程后续图片约 100–250 ms。

## 安装顺序

### 1. 创建 Python 3.11 环境

```powershell
py -3.11 -m venv E:\venvs\lpr311
E:\venvs\lpr311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2. 安装 CUDA 版 PyTorch

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

如果只使用 CPU：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装项目依赖

```powershell
cd C:\Users\HP\Desktop\license_plate_recognition
pip install -r requirements.txt
```

PaddleOCR 第一次运行会自动下载中文模型。当前模型缓存位于：

```text
C:\Users\HP\.paddleocr
```

## Windows 注意事项

1. PaddleOCR 2.9.1 必须使用 `numpy<2.0`，项目已固定为 NumPy 1.26.4。
2. OpenCV 的三种发行包必须保持版本一致，项目已统一为 4.10.0.84。
3. PaddleOCR 当前使用 CPU，YOLO 使用 GPU，避免 Windows 下 Paddle GPU 兼容问题。
4. 4GB 显存取 YOLOv8n 足够，但不适合大 batch 训练。