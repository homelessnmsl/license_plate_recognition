# PP-OCRv4 车牌微调

## 训练环境

- GPU 环境：`E:\venvs\paddle_gpu311`
- PaddlePaddle GPU：2.6.2
- PaddleOCR：2.9.1
- cuDNN：8.9.5
- GPU：RTX 3050 Laptop 4GB
- 训练代码：`E:\license_plate_data\PaddleOCR_train`
- 预训练权重：`E:\license_plate_data\pretrained\ch_PP-OCRv4_rec_train\student.pdparams`

## 数据

- 训练集：8,975 张车牌裁剪
- 验证集：1,579 张
- OCR 测试集：28 张
- 第 5 号人工检查过曝样本已排除
- OCR 训练集和验证集按文件内容去重
- 端到端测试集按内容哈希与训练集隔离

## 微调配置

| 参数 | 值 |
|---|---|
| 模型 | PP-OCRv4 SVTR_LCNet |
| 输入 | 3×48×320 |
| epoch | 3 |
| batch | 16 |
| 优化器 | Adam |
| 学习率 | 1e-4，Cosine |
| warmup | 1 epoch |
| 预训练 | student.pdparams |
| 设备 | GPU |

训练脚本：

```powershell
python scripts/train_ocr.py --epochs 3 --batch 16
```

## 结果

OCR 验证集：

- 整牌准确率：87.56%
- Norm Edit Distance：0.973

OCR 独立裁剪测试集：

- 整牌完全正确：26/28，92.86%
- 平均字符准确率：98.98%

30 张端到端测试图：

- 定位：30/30
- 整牌正确：28/30，93.3%
- 字符准确率：99.06%
- 接受识别：30/30
- 低质量拒识：0/30
- 平均处理时间：约802 ms

说明：拒识机制仍然保留。当前新测试集中的图片质量足够，因此没有触发拒识；以后遇到低置信度或格式异常时仍会返回“无法可靠识别”。

## 模型路径

```text
E:\license_plate_data\ocr_train_output\inference
```

项目通过环境变量使用该模型：

```powershell
$env:LPR_OCR_REC_MODEL_DIR = "E:\license_plate_data\ocr_train_output\inference"
```

`activate_env.ps1` 已自动设置该变量。

## 关键结论

- 3 轮微调后，OCR 验证整牌准确率从通用模型水平提升到 87.56%
- 端到端整体准确率达到 93.3%，满足课程优秀目标
- 剩余错误主要是单字符混淆，不是 YOLO 漏检
- 不应继续增加 YOLO 轮数来解决 OCR 字符错误