# PaddleOCR 微调数据

## 数据生成

脚本：

```powershell
E:\venvs\lpr311\Scripts\python.exe scripts\prepare_ocr_dataset.py `
  --yolo-root E:\license_plate_data\yolo `
  --output E:\license_plate_data\ocr `
  --test-root data\test
```

处理流程：

1. 读取 YOLO 训练集的真实检测框
2. 根据 CCPD 文件名解析车牌文本
3. 用颜色掩膜和最小外接矩形做透视矫正
4. 检查尺寸、长宽比、亮度、模糊度和文本合法性
5. 按文件哈希去除重复图片
6. 生成 PaddleOCR 的 `image_path<TAB>plate_text` 标签

## 数据规模

```text
有效裁剪：10,555
训练集：8,976
验证集：1,579
测试集：27
```

过滤情况：

| 原因 | 数量 |
|---|---:|
| 重复图片 | 81 |
| 长宽比异常 | 1000 |
| 过暗 | 57 |
| 过模糊 | 7 |
| 测试集长宽比异常 | 3 |

其中 9,212 张由颜色四边形成功透视矫正。

## 目录

```text
E:\license_plate_data\ocr\
├── images\
│   ├── train\
│   ├── val\
│   └── test\
├── labels\
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── ppocr_keys_v1.txt
└── manifest.json
```

标签格式：

```text
images/train/xxx.jpg	皖A12345
```

## 学习重点

- OCR 训练质量首先取决于裁剪和标签
- 车牌文本是强约束序列，不能使用左右翻转增强
- 训练集和验证集必须去重
- 过曝、严重模糊和角度过大的样本会引入噪声
- OCR 微调只解决字符识别问题，不解决 YOLO 漏检