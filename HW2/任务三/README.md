# U-Net图像分割 - 任务三

## 项目结构

task3_unet_segmentation/
├── config.py          # 配置文件
├── models.py          # U-Net模型（手写，无预训练）
├── dataset.py         # 数据集加载
├── losses.py          # 三种损失函数（CE/Dice/Combined）
├── metrics.py         # 评估指标（mIoU/Accuracy）
├── train.py           # 训练逻辑（含SwanLab）
├── main.py            # 主程序，三种损失对比
├── visualize.py       # 可视化工具
├── requirements.txt   # 依赖包
└── results/           # 结果保存目录

## 快速开始

1. 安装依赖：pip install -r requirements.txt

2. 准备数据：将Stanford Background Dataset放入 hw_2_3/StanfordBackgroundDataset/
   - images/ 目录存放 .jpg 图像
   - labels/ 目录存放 .txt 标签文件

3. 修改 config.py 中的 DATA_ROOT 为实际路径

4. 运行训练：python main.py

## 实验设置

| 参数 | 值 |
|------|-----|
| 数据集 | Stanford Background Dataset |
| 训练/验证划分 | 80% / 20% |
| 图像尺寸 | 256×256 |
| Batch Size | 8 |
| Epochs | 50 |
| 学习率 | 1e-4 |
| 优化器 | Adam |
| 损失函数 | Cross-Entropy / Dice / Combined |
| 评价指标 | mIoU, Pixel Accuracy |

## 实验结果

| 损失函数 | 最佳mIoU | 最佳准确率 |
|---------|---------|-----------|
| Cross-Entropy | 96.43% | 96.88% |
| Dice Loss | 96.69% | 97.11% |
| Combined Loss | 96.33% | 96.79% |

## 输出文件

- results/models/best_model_*.pth - 最佳模型权重
- results/curves/training_curves.png - 训练曲线图

## 依赖包

torch>=1.9.0, torchvision>=0.10.0, numpy>=1.19.0, Pillow>=8.0.0, matplotlib>=3.3.0, tqdm>=4.50.0, swanlab>=0.3.0

