"""
YOLOv8 模型训练脚本（使用 SwanLab 可视化）
满足任务二要求：
- 微调预训练模型
- SwanLab 可视化（训练/验证 loss 曲线 + mAP 曲线）
- 超参数分析
- 预训练消融实验（可选）
"""

import torch
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback
import swanlab


def train_model():
    """主训练函数"""
    
    # 检查 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========== 1. 初始化 SwanLab ==========
    swanlab.init(
        project="vehicle-detection-task2",
        experiment_name="yolov8m-vehicle-detection",
        description="YOLOv8m fine-tuning on RoadVehicleImagesDataset",
        config={
            "model": "yolov8m.pt",
            "epochs": 100,
            "batch_size": 8,
            "imgsz": 640,
            "learning_rate": 0.005,
            "dataset": "RoadVehicleImagesDataset",
            "classes": 21,
        }
    )
    
    # ========== 2. 加载预训练模型 ==========
    model = YOLO('yolov8m.pt')
    
    # ========== 3. 添加 SwanLab 回调（自动记录指标）==========
    add_swanlab_callback(model)
    
    # ========== 4. 训练 ==========
    print("\n开始训练...")
    model.train(
        data='config/dataset.yaml',
        epochs=100,
        batch=8,
        imgsz=640,
        device=device,
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        amp=False,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        project='runs/train',
        name='vehicle_detection_v2',
        exist_ok=True,
        workers=8,
    )
    
    # ========== 5. 验证并获取指标 ==========
    print("\n验证最佳模型...")
    metrics = model.val(data='config/dataset.yaml')
    
    # ========== 6. 输出结果 ==========
    print(f"\n{'='*50}")
    print(f"✅ 训练完成！")
    print(f"{'='*50}")
    print(f"最佳 mAP50:     {metrics.box.map50:.4f}")
    print(f"最佳 mAP50-95:  {metrics.box.map:.4f}")
    print(f"最佳 Precision: {metrics.box.mp:.4f}")
    print(f"最佳 Recall:    {metrics.box.mr:.4f}")
    print(f"\n模型保存位置:")
    print(f"  runs/train/vehicle_detection_v2/weights/best.pt")
    print(f"  runs/train/vehicle_detection_v2/weights/last.pt")
    print(f"{'='*50}")
    print(f"🔗 查看 SwanLab 可视化: https://swanlab.cn")
    print(f"{'='*50}")
    
    # ========== 7. 关闭 SwanLab（回调已记录所有曲线，无需再调用 log）==========
    swanlab.finish()
    
    return metrics


def validate_model(weights_path='runs/train/vehicle_detection_v2/weights/best.pt'):
    """单独验证模型（不训练）"""
    print(f"\n验证模型: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data='config/dataset.yaml')
    
    print(f"\n{'='*50}")
    print(f"验证结果")
    print(f"{'='*50}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"{'='*50}")
    
    return metrics


# ========== 可选：预训练消融实验 ==========
def train_from_scratch():
    """从零开始训练（不使用预训练权重）- 用于消融实验对比"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("训练模式: 从零开始（无预训练权重）...")
    
    swanlab.init(
        project="vehicle-detection-task2",
        experiment_name="yolov8m-from-scratch",
        description="Ablation study: training from scratch",
        config={
            "model": "yolov8m",
            "pretrained": False,
            "epochs": 100,
            "batch_size": 8,
        }
    )
    
    model = YOLO('yolov8m.yaml')
    add_swanlab_callback(model)
    
    model.train(
        data='config/dataset.yaml',
        epochs=100,
        batch=8,
        imgsz=640,
        device=device,
        lr0=0.01,
        amp=False,
        plots=True,
        project='runs/train',
        name='from_scratch',
        exist_ok=True,
    )
    
    metrics = model.val(data='config/dataset.yaml')
    print(f"\n✅ 从零训练完成！最佳 mAP50: {metrics.box.map50:.4f}")
    
    swanlab.finish()
    return metrics


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # 正常训练（使用预训练权重）
    train_model()
    
    # 可选：运行消融实验（从零开始训练）
    # 取消下面的注释即可运行对比实验
    # print("\n" + "="*50)
    # print("运行消融实验：从零开始训练（无预训练权重）")
    # print("="*50)
    # train_from_scratch()