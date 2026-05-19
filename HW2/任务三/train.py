# train.py - 完整修复版
"""
U-Net训练脚本
包含训练和验证逻辑，集成SwanLab可视化
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import swanlab

from losses import get_loss_function
from metrics import calculate_miou, calculate_accuracy


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_miou = 0
    total_acc = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_miou += calculate_miou(outputs, masks)
        total_acc += calculate_accuracy(outputs, masks)
        
        pbar.set_postfix({'loss': loss.item(), 'miou': total_miou/(pbar.n+1)})
    
    n_batches = len(train_loader)
    return (total_loss / n_batches, 
            total_miou / n_batches, 
            total_acc / n_batches)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_miou = 0
    total_acc = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_miou += calculate_miou(outputs, masks)
            total_acc += calculate_accuracy(outputs, masks)
    
    n_batches = len(val_loader)
    return (total_loss / n_batches, 
            total_miou / n_batches, 
            total_acc / n_batches)


def train_model(model, train_loader, val_loader, loss_type, 
                num_epochs, learning_rate, device, save_dir='results/models'):
    """
    完整训练流程，集成SwanLab
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取损失函数
    criterion = get_loss_function(loss_type)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 损失函数名称映射
    loss_names = {'ce': 'Cross-Entropy', 'dice': 'Dice Loss', 'combined': 'Combined Loss'}
    experiment_name = f"UNet_{loss_names[loss_type]}"
    
    # ========== 初始化SwanLab ==========
    swanlab.init(
        project="U-Net-Semantic-Segmentation",
        experiment_name=experiment_name,
        description=f"对比实验: {loss_names[loss_type]} - Stanford Background Dataset",
        config={
            "model": "UNet",
            "loss_function": loss_names[loss_type],
            "dataset": "Stanford Background Dataset",
            "train_ratio": 0.8,
            "image_size": "256x256",
            "batch_size": train_loader.batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
            "device": str(device)
        }
    )
    
    # 记录历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou': [],
        'train_acc': [], 'val_acc': [],
        'best_miou': 0, 'best_acc': 0, 'best_epoch': 0
    }
    
    print(f"\n开始训练: {loss_names[loss_type]}")
    print(f"SwanLab项目: U-Net-Semantic-Segmentation")
    print(f"实验名称: {experiment_name}")
    print(f"设备: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_loader.batch_size}")
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # 训练
        train_loss, train_miou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_miou, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"  ✓ 学习率降低: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # ========== SwanLab记录指标 ==========
        swanlab.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/mIoU": train_miou,
            "val/mIoU": val_miou,
            "train/accuracy": train_acc,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if val_miou > history['best_miou']:
            history['best_miou'] = val_miou
            history['best_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            
            model_path = os.path.join(save_dir, f'best_model_{loss_type}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': val_miou,
                'best_acc': val_acc,
                'loss_type': loss_type
            }, model_path)
            print(f"  ✓ 保存最佳模型 (mIoU: {val_miou:.4f})")
        
        # 打印结果
        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 打印训练完成总结
    print(f"\n训练完成: {loss_names[loss_type]}")
    print(f"最佳验证mIoU: {history['best_miou']:.4f} (Epoch {history['best_epoch']})")
    print(f"最佳验证准确率: {history['best_acc']:.4f}")
    
    # 结束SwanLab记录
    swanlab.finish()
    
    return history