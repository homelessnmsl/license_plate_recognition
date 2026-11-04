# 数据集规范

## 1. CCPD 训练数据

CCPD 文件名包含车牌区域、四个角点、车牌文本编码、亮度、倾斜和模糊程度。

转换命令：

```powershell
E:\venvs\lpr311\Scripts\python.exe scripts\prepare_ccpd_subset.py `
  --root E:\license_plate_data `
  --test-dir data\test

E:\venvs\lpr311\Scripts\python.exe scripts\prepare_ccpd.py `
  --ccpd-dir E:\license_plate_data\raw\ccpd_mixed_12k `
  --output E:\license_plate_data\yolo `
  --val-ratio 0.15 `
  --mode hardlink
```

输出结构：

```text
data/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── ground_truth.csv
configs/
└── plate_dataset.yaml
```

每个 YOLO 标注文件为：

```text
class_id x_center y_center width height
```

全部坐标归一化到 0–1。

## 2. 自建测试集

课程要求至少 20 张测试图片，建议准备 30 张：

- 正常白天
- 逆光
- 阴天/低照度
- 轻微倾斜
- 远距离
- 蓝牌、黄牌、绿牌
- 少量遮挡或反光

测试集必须独立于 CCPD 训练集。推荐存放：

```text
data/test/
```

标签文件格式：

```csv
image_path,plate_text,x1,y1,x2,y2
data/test/normal_01.jpg,京A12345,320,420,620,520
data/test/low_light_01.jpg,冀RDC3750,180,510,820,650
```

`x1,y1,x2,y2` 可以留空；但为空时只能评估识别结果，不能计算严格定位准确率。

## 3. 数据划分原则

- 训练集和验证集按图片划分，不能按裁剪块划分
- 同一辆车的连续视频帧不能同时出现在训练集和测试集
- 测试集一经确定不要反复用于调参
- 调参使用验证集，最终指标使用测试集
- 数据增强只能作用于训练集

## 4. 数据检查

训练前至少抽查 20 张：

- 框是否完整包住车牌
- 框是否包含过多车体背景
- 文本是否包含省份汉字
- 图像是否方向正确
- 标签文件是否与图片同名
- 训练集和验证集是否重复