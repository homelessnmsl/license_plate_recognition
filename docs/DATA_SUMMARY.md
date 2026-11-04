# 数据准备结果

## 数据源

- 训练候选集：`zenitsu09/ccpd-subset-30k`
- 独立测试集：`zenitsu09/ccpd-100k-yolo` 中的 `ccpd_test_1k.zip`
- 原始数据：CCPD，Xu et al., ECCV 2018
- Hugging Face 标注许可：训练集 MIT，测试集 Apache-2.0
- 镜像：`https://hf-mirror.com`

原始 ZIP 保存于：

```text
E:\license_plate_data\downloads
```

## 训练集

从 30,000 张中分层抽取 12,100 张：

| 类别 | 数量 |
|---|---:|
| ccpd_base | 4000 |
| ccpd_challenge | 1500 |
| ccpd_tilt | 1500 |
| ccpd_green | 1000 |
| ccpd_blur | 900 |
| ccpd_fn | 900 |
| ccpd_rotate | 500 |
| ccpd_db | 700 |
| ccpd_weather | 700 |
| ccpd_np | 400 |

`ccpd_np` 的 400 个文件没有 CCPD 文件名坐标/文本编码，已在转换阶段跳过，因此有效数据为 11,700 张。

YOLO 划分：

- 训练集：9,939 张
- 验证集：1,761 张
- 总有效图片：11,700 张
- 标签异常：0

转换后目录：

```text
E:\license_plate_data\yolo\
├── images\train
├── images\val
├── labels\train
├── labels\val
└── ground_truth.csv
```

数据集 YAML：

```text
configs/plate_dataset.yaml
```

## 独立测试集

从 1,000 张独立测试候选图中分层抽取 30 张：

- 蓝牌、绿牌
- 正常场景、倾斜、旋转
- 模糊、远距离
- 低照度、复杂天气
- 挑战场景

测试图片与训练子集无重叠。标签包含：

- 车牌文本
- 原始边界框 `x1,y1,x2,y2`

文件：

```text
data/test/ccpd_test_001.jpg ... ccpd_test_030.jpg
data/test/ground_truth.csv
```

测试图片本身因体积原因由 `.gitignore` 排除，标签表提交到 Git。

## 当前基线

使用旧 YOLO 权重进行流程验证，不代表最终训练结果：

- 定位准确率：53.3%
- 已定位样本的整牌识别率：100.0%
- 整体准确率：40.0%
- 字符准确率：48.4%

该结果说明旧权重对 CCPD 复杂场景泛化不足，需要重新训练 YOLOv8n。

## 数据总量

```text
E:\license_plate_data 约 3.11 GB
```

包含原始 ZIP、抽取图片、YOLO 数据和测试集。

## 复现命令

```powershell
E:\venvs\lpr311\Scripts\python.exe scripts\prepare_ccpd_subset.py `
  --root E:\license_plate_data `
  --test-dir data\test

E:\venvs\lpr311\Scripts\python.exe scripts\prepare_ccpd.py `
  --ccpd-dir E:\license_plate_data\raw\ccpd_mixed_12k `
  --output E:\license_plate_data\yolo `
  --val-ratio 0.15
```