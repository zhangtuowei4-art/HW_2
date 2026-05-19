# config.py
"""配置文件"""

# 数据配置
DATA_ROOT = "StanfordBackgroundDataset"  
TRAIN_RATIO = 0.8
IMAGE_SIZE = (256, 256)

# 训练配置
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda"

# 模型配置
N_CHANNELS = 3
N_CLASSES = 1
BILINEAR = True

# 其他
SEED = 42
SAVE_DIR = "results"