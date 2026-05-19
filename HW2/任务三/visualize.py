# visualize.py
"""
训练曲线可视化和预测结果可视化
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_curves(histories, save_dir='results/curves'):
    """
    绘制训练曲线
    包含: 训练损失、验证损失、mIoU、Accuracy
    """
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {'Cross-Entropy': '#1f77b4', 
              'Dice Loss': '#ff7f0e', 
              'Combined Loss': '#2ca02c'}
    linestyles = {'Cross-Entropy': '-', 
                  'Dice Loss': '--', 
                  'Combined Loss': '-.'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 训练损失
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['train_loss'], color=colors[name], 
                linestyle=linestyles[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 验证损失
    ax = axes[0, 1]
    for name, history in histories.items():
        ax.plot(history['val_loss'], color=colors[name], 
                linestyle=linestyles[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Validation Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 验证mIoU
    ax = axes[0, 2]
    for name, history in histories.items():
        ax.plot(history['val_miou'], color=colors[name], 
                linestyle=linestyles[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Validation mIoU', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 验证准确率
    ax = axes[1, 0]
    for name, history in histories.items():
        ax.plot(history['val_acc'], color=colors[name], 
                linestyle=linestyles[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 训练mIoU
    ax = axes[1, 1]
    for name, history in histories.items():
        ax.plot(history['train_miou'], color=colors[name], 
                linestyle=linestyles[name], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Training mIoU', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 最终结果对比柱状图
    ax = axes[1, 2]
    names = list(histories.keys())
    best_mious = [histories[name]['best_miou'] for name in names]
    bars = ax.bar(names, best_mious, color=[colors[n] for n in names])
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Best Validation mIoU', fontsize=14)
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, best_mious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_dir}/training_curves.png")


def plot_predictions(model, val_loader, device, epoch, save_dir='results/predictions'):
    """可视化模型预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # 获取一批数据
    images, masks = next(iter(val_loader))
    images = images[:4].to(device)
    masks = masks[:4].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    for i in range(4):
        # 原图
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image', fontsize=10)
        axes[i, 0].axis('off')
        
        # 真实标签
        axes[i, 1].imshow(masks[i].cpu().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        # 预测结果
        axes[i, 2].imshow(preds[i].cpu().squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}.png'), dpi=150)
    plt.close()


def print_experiment_summary(results):
    """打印实验总结"""
    print("\n" + "="*70)
    print("实验结果总结")
    print("="*70)
    
    print(f"\n{'损失函数':<20} {'最佳mIoU':<15} {'最佳准确率':<15} {'最佳Epoch':<12}")
    print("-"*65)
    
    for name, result in results.items():
        print(f"{name:<20} {result['best_miou']:.4f}          {result['best_acc']:.4f}          {result['best_epoch']}")
    
    print("\n" + "="*70)
    
    # 找出最佳损失函数
    best_loss = max(results.items(), key=lambda x: x[1]['best_miou'])
    print(f"\n最佳损失函数: {best_loss[0]} (mIoU: {best_loss[1]['best_miou']:.4f})")