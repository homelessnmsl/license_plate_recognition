# 自建测试集

将至少 20 张独立测试图片放在本目录，并在 `ground_truth.csv` 中填写车牌真值。

```csv
image_path,plate_text,x1,y1,x2,y2
normal_01.jpg,京A12345,320,420,620,520
low_light_01.jpg,冀RDC3750,180,510,820,650
```

如果暂时不填写定位框，可以只填写：

```csv
image_path,plate_text
normal_01.jpg,京A12345
```

测试图片不要来自训练集，否则整体准确率会被高估。