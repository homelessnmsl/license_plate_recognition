# Web 部署与内网穿透

本项目提供了一套可直接运行的 Web 推理服务：

```text
手机 / 电脑浏览器
        │  上传图片或调用手机摄像头
        ▼
FastAPI  ──► ONNX Runtime + YOLOv8n（车牌定位）
        └──► Paddle Inference + PP-OCRv4（车牌字符识别）
        ▼
标注图片 + JSON 结果
```

## 1. 模型产物

运行导出脚本：

```powershell
. .\activate_env.ps1
python scripts\export_models.py --device 0
```

导出后会生成：

```text
deploy/models/
├── yolov8n_plate.onnx       # 部署用 YOLO 检测模型
├── yolov8n_plate.pt         # 原始 PyTorch 权重备份
├── manifest.json            # 大小、SHA-256、验证指标
└── ocr_rec/
    ├── inference.pdmodel    # PP-OCRv4 微调识别模型
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.yml
```

YOLO 使用 ONNX 是为了减少 Web 服务对 PyTorch 推理路径的依赖；PaddleOCR 的微调模型本身已经是 Paddle Inference 部署格式，直接随服务打包。

## 2. 安装 Web 依赖

```powershell
. .\activate_env.ps1
pip install -r requirements-web.txt
```

`onnx==1.16.2`、`onnxruntime==1.19.2` 固定版本，避免新版依赖要求 `protobuf>=4` 而破坏 PaddlePaddle 2.6.2。

## 3. 启动局域网服务

```powershell
.\scripts\start_web.ps1 -Port 8000
```

脚本会自动：

- 使用 `E:\venvs\lpr311`
- 优先加载 `deploy/models/yolov8n_plate.onnx`
- 加载 `deploy/models/ocr_rec`
- 生成 8 位访问码
- 预热模型
- 打印本机局域网访问地址

同一 Wi-Fi 下的手机可直接打开终端输出的地址。若 Windows 防火墙拦截，请允许 Python 在“专用网络”上通信。

## 4. 启动公网隧道

保持 Web 服务窗口运行，另开一个 PowerShell：

```powershell
.\scripts\start_tunnel.ps1 -Port 8000
```

脚本使用 OpenSSH 连接 `localhost.run`，终端会输出类似：

```text
https://xxxxxxxxxxxx.lhr.life
```

把访问码附加到地址后面：

```text
https://xxxxxxxxxxxx.lhr.life/?key=启动服务时显示的访问码
```

手机扫码或直接打开该链接即可。浏览器会自动保存访问码，后续不必重复输入。

## 5. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | 响应式 Web 前端 |
| `GET` | `/api/info` | 模型状态、上传限制、是否启用访问码 |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/recognize` | 图片识别，`multipart/form-data` 字段名为 `file` |
| `GET` | `/api/docs` | FastAPI 交互文档 |

调用示例：

```powershell
curl.exe -X POST `
  -H "X-Access-Key: 你的访问码" `
  -F "file=@data/test/ccpd_test_002.jpg;type=image/jpeg" `
  http://127.0.0.1:8000/api/recognize
```

响应中包含：

- `plates`：车牌文本、颜色、状态、检测/OCR/综合置信度、检测框、耗时
- `annotated_image`：标注后的 JPEG Data URL
- `elapsed_ms`：服务端处理耗时

## 6. 配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LPR_WEB_TOKEN` | 空 | 访问码；为空则关闭鉴权 |
| `LPR_MAX_UPLOAD_MB` | `15` | 单张图片大小上限 |
| `LPR_MAX_IMAGE_SIDE` | `2200` | 推理前最长边缩放上限 |
| `LPR_RATE_LIMIT` | `30` | 每 IP 每分钟请求数 |
| `LPR_WARMUP` | `true` | 启动时预热模型 |
| `LPR_ONNX_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | ONNX Runtime 执行提供者，自动降级到 CPU |

## 7. 验证结果

导出后的 ONNX 检测模型在 30 张隔离测试图上复测：

- 定位率：30/30
- 识别准确率：30/30
- 字符准确率：100%
- 预热后单张端到端耗时：约 0.22–0.77 秒（RTX 3050 笔记本，OCR 使用 CPU）

报告位于 `outputs/json/evaluation_onnx.json`，模型校验信息位于 `deploy/models/manifest.json`。

## 8. 安全与公网使用

- 服务默认限制 15 MB 上传、4000 万像素和每分钟 30 次请求。
- 图片在内存中处理，不落盘。
- 公网使用时务必设置 `LPR_WEB_TOKEN`；前端支持从 URL 的 `?key=` 参数读取访问码。
- `localhost.run` 免费隧道地址在重连后会变化；长期部署建议使用 Cloudflare Named Tunnel、frp 或 nginx + 固定域名。
- 公网演示链接不要长期公开传播，避免被无关请求占用推理资源。

## 9. 常见问题：本地能识别，公网返回 502

这通常不是模型或 FastAPI 的问题，而是免费隧道地址已失效。`localhost.run` 免费域名会定期更换，旧链接随后返回 `502 Bad Gateway`。

处理方法：

1. 保持 `start_web.ps1` 窗口运行。
2. 重新运行 `start_tunnel.ps1`，终端会打印新的 HTTPS 地址。
3. 在新地址后追加原来的 `?key=访问码`。

`start_tunnel.ps1` 已加入自动重连，但重连后的公网域名通常会变化。
