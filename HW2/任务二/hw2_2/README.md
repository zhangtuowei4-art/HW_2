# 任务二：场景目标检测与视频多目标跟踪

## 项目简介

基于 YOLOv8 + ByteTrack 的道路车辆检测与多目标跟踪系统。

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows


# 安装依赖
pip install torch>=2.0.0 torchvision>=0.15.0
pip install ultralytics>=8.0.0 opencv-python>=4.8.0 numpy>=1.24.0
pip install matplotlib>=3.7.0 pandas>=2.0.0 pyyaml>=6.0
pip install swanlab

或

pip install -r requirements.txt
```


## 项目结构

```
hw2_2/
│  
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── track.py              # 跟踪脚本
│   ├── line_counter.py       # 计数模块
│   └── config/
│       ├── dataset.yaml      # 数据集配置
│       └── bytetrack.yaml    # 跟踪器配置
├── runs/train/vehicle_detection_v2/weights/
│   └── best.pt   # 训练好的模型
├── videos/
│   ├── input/                # 输入视频
│   └── output/               # 输出视频
├── data/                     # 数据集
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
└── README.md
```

## 快速使用

### 训练模型

```bash
python scripts/train.py
```

### 视频跟踪

```bash
# 基本跟踪
python scripts/track.py --source videos/input/test.mp4

# 带计数线
python scripts/track.py --source videos/input/test.mp4 --line "200,350,1080,350"

# 保存结果
python scripts/track.py --source videos/input/test.mp4 --output videos/output/result.mp4 --line "200,350,1080,350"
```

### 摄像头实时

```bash
python scripts/track.py --source 0 --line "200,350,1080,350"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 视频路径或摄像头ID | 必填 |
| `--model` | 模型权重路径 | `runs/train/vehicle_detection_v2/weights/best.pt` |
| `--output` | 输出视频路径 | None |
| `--conf` | 置信度阈值 | 0.3 |
| `--line` | 计数线坐标 | None |

## 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | YOLOv8m |
| 输入尺寸 | 640×640 |
| 批次大小 | 8 |
| 学习率 | 0.005 |
| 优化器 | AdamW |

## 实验结果

| 指标 | 数值 |
|------|------|
| mAP50 | 0.561 |
| mAP50-95 | 0.309 |
| Precision | 0.671 |
| Recall | 0.457 |
| FPS | 102 |

## 模型下载

- [Google Drive] https://drive.google.com/drive/folders/xxx

## 可视化

SwanLab 项目：https://swanlab.cn/@chaiyk/vehicle-detection-task2

## 常见问题

### CUDA 不可用

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 找不到模型

```bash
python scripts/track.py --source video.mp4 --model /path/to/best.pt
```

## 参考资料

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [SwanLab](https://swanlab.cn)