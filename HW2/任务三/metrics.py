# metrics.py
"""
语义分割评估指标实现
"""

import torch


def calculate_miou(pred, target, threshold=0.5, num_classes=2):
    """
    计算mIoU (mean Intersection over Union)
    
    参数:
        pred: 模型输出 (batch_size, 1, H, W)
        target: 标签 (batch_size, 1, H, W)
        threshold: 二值化阈值
        num_classes: 类别数（2表示二分类）
    
    返回:
        miou: 平均IoU值
    """
    # 二值化预测
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # 计算二分类的IoU
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()


def calculate_accuracy(pred, target, threshold=0.5):
    """
    计算像素准确率 (Pixel Accuracy)
    
    参数:
        pred: 模型输出
        target: 标签
        threshold: 二值化阈值
    
    返回:
        accuracy: 像素准确率
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    correct = (pred_binary == target).float().sum()
    total = target.numel()
    
    return (correct / total).item()


def calculate_precision_recall_f1(pred, target, threshold=0.5):
    """
    计算精确率、召回率和F1分数
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    tp = ((pred_binary == 1) & (target == 1)).float().sum()
    fp = ((pred_binary == 1) & (target == 0)).float().sum()
    fn = ((pred_binary == 0) & (target == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()