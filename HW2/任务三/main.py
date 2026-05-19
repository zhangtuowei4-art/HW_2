# main.py
"""主程序入口"""
import torch
import os
from torch.utils.data import DataLoader, random_split

import config
from models import UNet, init_weights
from train import train_model
from visualize import plot_training_curves, print_experiment_summary


# main.py 中修改数据集加载部分
from dataset import StanfordBackgroundDataset, get_transforms, check_dataset_structure

def get_data_loaders():
    """获取数据加载器"""
    
    # 先检查数据结构
    check_dataset_structure(config.DATA_ROOT)
    
    # 获取变换
    img_transform, mask_transform = get_transforms(config.IMAGE_SIZE)
    
    # 创建数据集（使用regions标签，它通常包含完整的语义分割信息）
    full_dataset = StanfordBackgroundDataset(
        root_dir=config.DATA_ROOT,
        transform=img_transform,
        target_transform=mask_transform,
        binary=True,           # 二值化为前景/背景
        label_type='regions'   # 使用regions.txt标签
    )
    
    # 划分训练集和验证集
    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=True, num_workers=0, pin_memory=True  # num_workers设为0避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, 
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"\n数据集信息:")
    print(f"  总数: {len(full_dataset)}")
    print(f"  训练集: {len(train_dataset)} ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"  验证集: {len(val_dataset)} ({(1-config.TRAIN_RATIO)*100:.0f}%)")
    
    return train_loader, val_loader

def main():
    print("="*60)
    print("任务三: U-Net图像分割 - 三种损失函数对比实验")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_loader, val_loader = get_data_loaders()
    results = {}
    
    # 对三种损失函数分别训练
    for loss_type in ['ce', 'dice', 'combined']:
        print(f"\n训练: {loss_type.upper()}")
        
        model = UNet().to(device)
        model = init_weights(model)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_type=loss_type,
            num_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            device=device
        )
        
        loss_names = {'ce': 'Cross-Entropy', 'dice': 'Dice Loss', 'combined': 'Combined Loss'}
        results[loss_names[loss_type]] = history
    
    # 绘制对比曲线
    plot_training_curves(results)
    print_experiment_summary(results)
    print("\n完成!")


if __name__ == "__main__":
    main()