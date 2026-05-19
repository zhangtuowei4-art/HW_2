# check_labels.py - 修复版
"""
检查数据集标签分布
"""
import numpy as np
import torch
from dataset import StanfordBackgroundDataset, get_transforms

# 不使用变换，直接读取原始数据
def check_labels():
    print("="*50)
    print("检查Stanford Background Dataset标签")
    print("="*50)
    
    # 创建数据集（不二值化，不使用变换）
    dataset = StanfordBackgroundDataset(
        root_dir="StanfordBackgroundDataset",  # 修改为你的路径
        binary=False,  # 不二值化，查看原始类别
        label_type='regions',
        transform=None,
        target_transform=None
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 检查前5个样本的标签分布
    all_classes = set()
    
    for i in range(min(5, len(dataset))):
        _, mask = dataset[i]
        
        # 转换为numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        unique_vals = np.unique(mask_np)
        class_counts = {val: np.sum(mask_np == val) for val in unique_vals}
        
        print(f"\n样本 {i}: {dataset.images[i]}")
        print(f"  标签形状: {mask_np.shape}")
        print(f"  唯一值: {unique_vals}")
        print(f"  类别分布: {class_counts}")
        
        all_classes.update(unique_vals)
    
    print(f"\n所有出现过的类别: {sorted(all_classes)}")
    
    # 检查类别含义
    print("\n类别说明:")
    print("  0: 背景")
    print("  1-8: 不同物体类别 (天空、树木、道路、建筑物等)")
    print("  9-18: 细粒度类别")
    
    # 统计整个数据集的类别分布（可选，会比较慢）
    print("\n是否统计全部数据集？(可能会很慢)")
    all_class_counts = {}
    for i in range(min(50, len(dataset))):  # 只统计50张作为示例
        _, mask = dataset[i]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        for val in np.unique(mask_np):
            all_class_counts[val] = all_class_counts.get(val, 0) + np.sum(mask_np == val)
    
    print(f"采样统计(50张): {all_class_counts}")


if __name__ == "__main__":
    check_labels()