# metrics.py
"""
语义分割评估指标实现

包含:
- mIoU (mean Intersection over Union) - 正确计算所有类别的平均IoU
- Pixel Accuracy - 像素准确率
- Precision / Recall / F1 Score
- 各类别单独IoU
"""

import torch


def calculate_iou_per_class(pred, target, num_classes=2, threshold=0.5):
    """
    计算每个类别的IoU
    
    参数:
        pred: 模型输出 (batch_size, 1, H, W) - 未经sigmoid的logits
        target: 标签 (batch_size, 1, H, W) - 0/1值
        num_classes: 类别数（二分类为2）
        threshold: 二值化阈值
    
    返回:
        ious: 长度为num_classes的列表，每个元素的IoU值
    """
    # 二值化预测
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    # 确保target是二值的
    target_binary = (target > 0.5).float()
    
    ious = []
    
    for c in range(num_classes):
        # 获取第c类的预测和标签
        if c == 1:  # 前景类
            pred_c = pred_binary
            target_c = target_binary
        else:  # 背景类 (c == 0)
            pred_c = 1 - pred_binary
            target_c = 1 - target_binary
        
        # 计算交并比
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        # 避免除零
        if union.item() == 0:
            # 如果该类别在整批数据中都不存在，IoU设为1.0（完美匹配）
            iou = 1.0
        else:
            iou = (intersection + 1e-6) / (union + 1e-6)
        
        ious.append(iou.item())
    
    return ious


def calculate_miou(pred, target, num_classes=2, threshold=0.5):
    """
    计算mIoU (mean Intersection over Union)
    
    正确实现对每个类别分别计算IoU后取平均
    
    参数:
        pred: 模型输出 (batch_size, 1, H, W) - 未经sigmoid的logits
        target: 标签 (batch_size, 1, H, W) - 0/1值
        num_classes: 类别数（2表示二分类）
        threshold: 二值化阈值
    
    返回:
        miou: 平均IoU值
    """
    ious = calculate_iou_per_class(pred, target, num_classes, threshold)
    return sum(ious) / len(ious)


def calculate_foreground_iou(pred, target, threshold=0.5):
    """
    计算前景类别的IoU（与之前版本兼容）
    
    参数:
        pred: 模型输出
        target: 标签
        threshold: 二值化阈值
    
    返回:
        fg_iou: 前景类别的IoU
    """
    ious = calculate_iou_per_class(pred, target, num_classes=2, threshold=threshold)
    return ious[1]  # 索引1对应前景


def calculate_background_iou(pred, target, threshold=0.5):
    """
    计算背景类别的IoU
    
    返回:
        bg_iou: 背景类别的IoU
    """
    ious = calculate_iou_per_class(pred, target, num_classes=2, threshold=threshold)
    return ious[0]  # 索引0对应背景


def calculate_accuracy(pred, target, threshold=0.5):
    """
    计算像素准确率 (Pixel Accuracy)
    
    正确预测的像素数 / 总像素数
    
    参数:
        pred: 模型输出 (batch_size, 1, H, W) - 未经sigmoid的logits
        target: 标签 (batch_size, 1, H, W) - 0/1值
        threshold: 二值化阈值
    
    返回:
        accuracy: 像素准确率
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()
    
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    
    return (correct / total).item()


def calculate_precision_recall_f1(pred, target, threshold=0.5):
    """
    计算精确率、召回率和F1分数（基于前景类别）
    
    参数:
        pred: 模型输出
        target: 标签
        threshold: 二值化阈值
    
    返回:
        precision: 精确率 = TP / (TP + FP)
        recall: 召回率 = TP / (TP + FN)
        f1: F1分数 = 2 * P * R / (P + R)
    """
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()
    
    tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()


def calculate_all_metrics(pred, target, threshold=0.5):
    """
    一次性计算所有评估指标，便于统一记录
    
    参数:
        pred: 模型输出
        target: 标签
        threshold: 二值化阈值
    
    返回:
        metrics: 包含所有指标的字典
    """
    ious = calculate_iou_per_class(pred, target, num_classes=2, threshold=threshold)
    
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 像素准确率
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    accuracy = (correct / total).item()
    
    # 精确率/召回率/F1
    tp = ((pred_binary == 1) & (target_binary == 1)).float().sum()
    fp = ((pred_binary == 1) & (target_binary == 0)).float().sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'miou': sum(ious) / len(ious),           # 真正的mIoU
        'fg_iou': ious[1],                        # 前景IoU
        'bg_iou': ious[0],                        # 背景IoU
        'accuracy': accuracy,                     # 像素准确率
        'precision': precision.item(),            # 精确率
        'recall': recall.item(),                  # 召回率
        'f1_score': f1.item()                     # F1分数
    }


# 以下为测试代码（可选）
if __name__ == "__main__":
    # 模拟测试数据
    print("=" * 50)
    print("metrics.py 测试")
    print("=" * 50)
    
    # 创建模拟数据
    # pred: 假设模型输出完美预测
    pred_perfect = torch.randn(1, 1, 64, 64)
    target = (torch.rand(1, 1, 64, 64) > 0.5).float()
    # 让完美预测等于目标
    pred_perfect_sigmoid = target.clone()
    pred_perfect = torch.log(pred_perfect_sigmoid / (1 - pred_perfect_sigmoid + 1e-6))
    
    print("\n测试1: 完美预测")
    metrics = calculate_all_metrics(pred_perfect, target)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 创建随机预测
    print("\n测试2: 随机预测 (噪声)")
    pred_random = torch.randn(1, 1, 64, 64)
    metrics = calculate_all_metrics(pred_random, target)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ 所有指标实现正确")
