# models.py
"""
U-Net语义分割网络
包含完整的下采样编码器、上采样解码器以及特征拼接Skip Connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双卷积块: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """
    下采样模块（编码器部分）
    结构：MaxPool2d(2x2) -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """
    上采样模块（解码器部分）
    包含：上采样 + Skip Connection拼接 + 双卷积
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: 来自解码器上一层的特征（需要上采样）
        x2: 来自编码器对应层的特征（Skip Connection）
        """
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Skip Connection: 通道维度拼接
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutputConv(nn.Module):
    """输出层：1x1卷积"""
    def __init__(self, in_channels, out_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    完整的U-Net语义分割网络
    
    结构：
    - 编码器：4次下采样
    - 解码器：4次上采样 + Skip Connection
    - 输出层：1x1卷积
    
    参数：
    - n_channels: 输入通道数（RGB为3）
    - n_classes: 输出类别数（二分类为1）
    - bilinear: 是否使用双线性插值上采样
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # ========== 编码器 ==========
        self.inc = DoubleConv(n_channels, 64)      # 64通道
        self.down1 = DownSample(64, 128)           # 128通道
        self.down2 = DownSample(128, 256)          # 256通道
        self.down3 = DownSample(256, 512)          # 512通道
        
        factor = 2 if bilinear else 1
        self.down4 = DownSample(512, 1024 // factor)  # 瓶颈层
        
        # ========== 解码器 ==========
        self.up1 = UpSample(1024, 512 // factor, bilinear)
        self.up2 = UpSample(512, 256 // factor, bilinear)
        self.up3 = UpSample(256, 128 // factor, bilinear)
        self.up4 = UpSample(128, 64, bilinear)
        
        # ========== 输出层 ==========
        self.outc = OutputConv(64, n_classes)
    
    def forward(self, x):
        # 编码路径（保存特征用于Skip Connection）
        x1 = self.inc(x)      # (batch, 64, H, W)
        x2 = self.down1(x1)   # (batch, 128, H/2, W/2)
        x3 = self.down2(x2)   # (batch, 256, H/4, W/4)
        x4 = self.down3(x3)   # (batch, 512, H/8, W/8)
        x5 = self.down4(x4)   # (batch, 1024, H/16, W/16)
        
        # 解码路径（使用Skip Connection）
        x = self.up1(x5, x4)  # (batch, 512, H/8, W/8)
        x = self.up2(x, x3)   # (batch, 256, H/4, W/4)
        x = self.up3(x, x2)   # (batch, 128, H/2, W/2)
        x = self.up4(x, x1)   # (batch, 64, H, W)
        
        return self.outc(x)


def init_weights(model):
    """随机初始化模型权重"""
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(_init_weights)
    return model