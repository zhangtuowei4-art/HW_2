# losses.py
"""
手写实现的三种损失函数：
1. 标准交叉熵损失
2. Dice Loss
3. 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss - 解决前景/背景像素不平衡问题
    
    Dice = 2 * |A ∩ B| / (|A| + |B| + smooth)
    Dice Loss = 1 - Dice
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        参数:
            pred: 模型输出 (batch_size, 1, H, W) - 未经过sigmoid
            target: 标签 (batch_size, 1, H, W) - 0/1值
        返回:
            dice_loss: 标量
        """
        # 应用sigmoid将输出映射到[0,1]
        pred_sigmoid = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    组合损失: 交叉熵损失 + Dice Loss
    
    结合两者优点：
    - CE Loss: 梯度稳定，收敛快速
    - Dice Loss: 直接优化IoU，处理类别不平衡
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


def get_loss_function(loss_type, bce_weight=0.5, dice_weight=0.5):
    """
    根据类型获取损失函数
    
    参数:
        loss_type: 'ce', 'dice', 'combined'
    返回:
        criterion: 损失函数实例
    """
    if loss_type == 'ce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss(bce_weight, dice_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")