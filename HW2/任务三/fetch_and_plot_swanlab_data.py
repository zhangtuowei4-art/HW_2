# fetch_and_plot_swanlab_data.py - 最终修复版
import swanlab
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 👇 请将这里换成你自己的信息 👇 ====================
USERNAME = "chaiyk"
PROJECT_NAME = "U-Net-Semantic-Segmentation"

EXP_IDS = {
    "Cross-Entropy": "7olwpwd9id6mafwqyxa1q",
    "Dice Loss": "gqyfw5vqzr8xuzvly563i",
    "Combined Loss": "rtl9osz3htbbkw7k12qyk"
}

API_KEY = "K1rvZBbg0XpCpC7fsYDVj"  # 请替换为你的实际 API Key
# ===================================================================


def fetch_metric_data(exp_id, metric_key, api_key):
    """通过 SwanLab API 获取指定实验的指标数据"""
    print(f"正在获取实验 {exp_id} 的指标: {metric_key}...")
    try:
        api = swanlab.Api(api_key=api_key)
        run_path = f"{USERNAME}/{PROJECT_NAME}/{exp_id}"
        run = api.run(path=run_path)
        
        metrics_df = run.metrics(keys=[metric_key])
        
        if metrics_df is None or metrics_df.empty:
            print(f"  警告: 未找到指标 {metric_key}")
            return pd.DataFrame()
        
        print(f"  成功获取 {len(metrics_df)} 条数据。")
        return metrics_df
    except Exception as e:
        print(f"  获取实验 {exp_id} 数据失败: {e}")
        return pd.DataFrame()


def extract_values(df, metric_key):
    """
    从 DataFrame 中提取 step (索引) 和 value
    SwanLab 返回的 DataFrame 格式:
    - 索引: step 序号 (0, 1, 2, ...)
    - 列: metric_key 的值
    """
    if df.empty:
        return [], []
    
    # step 使用索引
    steps = df.index.tolist()
    # 值使用 metric_key 这一列
    values = df[metric_key].tolist()
    
    return steps, values


def plot_loss_curves(all_loss_data, save_path="loss_curves.png"):
    """绘制训练/验证损失曲线"""
    plt.figure(figsize=(12, 6))
    
    colors = {'Cross-Entropy': '#1f77b4', 'Dice Loss': '#2ca02c', 'Combined Loss': '#d62728'}
    linestyles = {'train': '-', 'val': '--'}
    
    for exp_name, exp_data in all_loss_data.items():
        # 绘制训练损失
        train_df = exp_data.get('train/loss', pd.DataFrame())
        if not train_df.empty:
            steps, values = extract_values(train_df, 'train/loss')
            plt.plot(steps, values, 
                    color=colors[exp_name], 
                    linestyle=linestyles['train'],
                    linewidth=1.5,
                    label=f"{exp_name} (Train)")
        
        # 绘制验证损失
        val_df = exp_data.get('val/loss', pd.DataFrame())
        if not val_df.empty:
            steps, values = extract_values(val_df, 'val/loss')
            plt.plot(steps, values, 
                    color=colors[exp_name], 
                    linestyle=linestyles['val'],
                    linewidth=1.5,
                    label=f"{exp_name} (Val)")
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss Curves', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"损失曲线已保存至: {save_path}")


def plot_accuracy_curves(all_acc_data, save_path="accuracy_curves.png"):
    """绘制验证集准确率曲线"""
    plt.figure(figsize=(12, 6))
    
    colors = {'Cross-Entropy': '#1f77b4', 'Dice Loss': '#2ca02c', 'Combined Loss': '#d62728'}
    
    for exp_name, acc_df in all_acc_data.items():
        if acc_df is not None and not acc_df.empty:
            steps, values = extract_values(acc_df, 'val/accuracy')
            plt.plot(steps, values, 
                    color=colors[exp_name], 
                    linewidth=2,
                    marker='o',
                    markersize=3,
                    label=exp_name)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.90, 1.0])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"准确率曲线已保存至: {save_path}")


def plot_miou_curves(all_miou_data, save_path="miou_curves.png"):
    """绘制验证集 mIoU 曲线"""
    plt.figure(figsize=(12, 6))
    
    colors = {'Cross-Entropy': '#1f77b4', 'Dice Loss': '#2ca02c', 'Combined Loss': '#d62728'}
    
    for exp_name, miou_df in all_miou_data.items():
        if miou_df is not None and not miou_df.empty:
            steps, values = extract_values(miou_df, 'val/mIoU')
            plt.plot(steps, values, 
                    color=colors[exp_name], 
                    linewidth=2,
                    marker='o',
                    markersize=3,
                    label=exp_name)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation mIoU', fontsize=12)
    plt.title('Validation mIoU Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.90, 0.98])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"mIoU曲线已保存至: {save_path}")


def plot_combined_curves(all_loss_data, all_acc_data, all_miou_data, save_path="combined_curves.png"):
    """绘制组合图：2x2 布局，包含损失、准确率、mIoU"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'Cross-Entropy': '#1f77b4', 'Dice Loss': '#2ca02c', 'Combined Loss': '#d62728'}
    linestyles = {'train': '-', 'val': '--'}
    
    # 1. 损失曲线
    ax = axes[0, 0]
    for exp_name, exp_data in all_loss_data.items():
        train_df = exp_data.get('train/loss', pd.DataFrame())
        if not train_df.empty:
            steps, values = extract_values(train_df, 'train/loss')
            ax.plot(steps, values, color=colors[exp_name], linestyle='-', 
                   linewidth=1.5, label=f"{exp_name} (Train)")
        
        val_df = exp_data.get('val/loss', pd.DataFrame())
        if not val_df.empty:
            steps, values = extract_values(val_df, 'val/loss')
            ax.plot(steps, values, color=colors[exp_name], linestyle='--', 
                   linewidth=1.5, label=f"{exp_name} (Val)")
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Loss Curves', fontsize=13)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. mIoU曲线
    ax = axes[0, 1]
    for exp_name, miou_df in all_miou_data.items():
        if not miou_df.empty:
            steps, values = extract_values(miou_df, 'val/mIoU')
            ax.plot(steps, values, color=colors[exp_name], linewidth=2, 
                   marker='o', markersize=3, label=exp_name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('mIoU', fontsize=11)
    ax.set_title('Validation mIoU Curves', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.90, 0.98])
    
    # 3. 准确率曲线
    ax = axes[1, 0]
    for exp_name, acc_df in all_acc_data.items():
        if not acc_df.empty:
            steps, values = extract_values(acc_df, 'val/accuracy')
            ax.plot(steps, values, color=colors[exp_name], linewidth=2, 
                   marker='o', markersize=3, label=exp_name)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy Curves', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.90, 1.0])
    
    # 4. 最终结果对比柱状图
    ax = axes[1, 1]
    exp_names = list(all_miou_data.keys())
    best_mious = []
    best_accs = []
    for exp_name in exp_names:
        miou_df = all_miou_data.get(exp_name, pd.DataFrame())
        acc_df = all_acc_data.get(exp_name, pd.DataFrame())
        if not miou_df.empty:
            best_mious.append(max(extract_values(miou_df, 'val/mIoU')[1]))
        else:
            best_mious.append(0)
        if not acc_df.empty:
            best_accs.append(max(extract_values(acc_df, 'val/accuracy')[1]))
        else:
            best_accs.append(0)
    
    x = np.arange(len(exp_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, best_mious, width, label='Best mIoU', color='#1f77b4')
    bars2 = ax.bar(x + width/2, best_accs, width, label='Best Accuracy', color='#ff7f0e')
    ax.set_xlabel('Loss Function', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Best Results Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, best_mious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"组合曲线图已保存至: {save_path}")


def main():
    print("="*60)
    print("SwanLab 数据获取与绘图工具")
    print("="*60)
    
    all_loss_data = {}
    all_acc_data = {}
    all_miou_data = {}
    
    metric_keys = {
        'loss': ['train/loss', 'val/loss'],
        'accuracy': ['val/accuracy'],
        'miou': ['val/mIoU']
    }
    
    for exp_name, exp_id in EXP_IDS.items():
        print(f"\n正在处理实验: {exp_name}")
        print("-"*40)
        
        loss_data = {}
        for loss_key in metric_keys['loss']:
            df = fetch_metric_data(exp_id, loss_key, API_KEY)
            if not df.empty:
                loss_data[loss_key] = df
        all_loss_data[exp_name] = loss_data
        
        acc_df = fetch_metric_data(exp_id, metric_keys['accuracy'][0], API_KEY)
        all_acc_data[exp_name] = acc_df
        
        miou_df = fetch_metric_data(exp_id, metric_keys['miou'][0], API_KEY)
        all_miou_data[exp_name] = miou_df
    
    print("\n" + "="*60)
    print("开始生成图表...")
    print("="*60)
    
    # 生成各种图表
    plot_loss_curves(all_loss_data)
    plot_accuracy_curves(all_acc_data)
    plot_miou_curves(all_miou_data)
    plot_combined_curves(all_loss_data, all_acc_data, all_miou_data)
    
    print("\n所有图表生成完毕！")
    print("\n生成的文件:")
    print("  - loss_curves.png (损失曲线)")
    print("  - accuracy_curves.png (准确率曲线)")
    print("  - miou_curves.png (mIoU曲线)")
    print("  - combined_curves.png (组合图)")


if __name__ == "__main__":
    main()