本项目是一个基于 PyTorch 的牛津 102 类花卉图像分类微调框架，包含两个核心模块：

数据处理 (dataset.py)：解析 .mat 格式的 102 Flowers 数据集，提供带 ImageNet 标准化的训练（含随机裁剪/翻转/颜色抖动）与验证测试数据加载。 模型训练 (train_baseline.py)：提供完整的训练与评估流程，核心特性包括： 多模型支持：ResNet18/34、SE-ResNet 及 ViT/Swin（通过 timm）。 进阶训练策略：支持分层学习率、Backbone 冻结、Warmup 及 Cosine 衰减。 评估与追踪：计算 Loss、Acc 及 mAP 指标，集成 SwanLab 实验记录，支持命令行参数灵活配置。

实验运行策略：

实验1
python train_baseline.py --exp_name "1_Baseline" --model_name resnet18 --pretrained --freeze_epochs 5 --epochs 30 --batch_size 32 --lr_fc 1e-3 --lr_backbone 1e-4

实验2
python train_baseline.py --exp_name "2_Ablation_NoPretrained" --model_name resnet18 --no-pretrained --epochs 30 --batch_size 32 --lr_fc 1e-3

实验3
python train_baseline.py --exp_name "3_SE_ResNet18" --model_name resnet18 --pretrained --use_se --freeze_epochs 5 --epochs 30 --batch_size 32 --lr_fc 1e-3 --lr_se 1e-4 --lr_backbone 1e-4

实验4
python train_baseline.py --exp_name "4_Transformer_ViT" --model_name vit_tiny --pretrained --freeze_epochs 5 --epochs 30 --batch_size 32 --lr_fc 1e-3 --lr_backbone 1e-4

实验5
python train_baseline.py --exp_name "5_HighLR" --model_name resnet18 --pretrained --freeze_epochs 5 --epochs 30 --batch_size 32 --lr_fc 3e-3 --lr_backbone 3e-4

实验6
python train_baseline.py --exp_name "6_SmallBS_AdjLR" --model_name resnet18 --pretrained --freeze_epochs 5 --epochs 30 --batch_size 16 --lr_fc 2e-3 --lr_backbone 2e-4

实验7
python train_baseline.py --exp_name "7_LongTrain" --model_name resnet18 --pretrained --freeze_epochs 10 --epochs 50 --batch_size 32 --lr_fc 1e-3 --lr_backbone 1e-4

实验8
python train_baseline.py --exp_name "8_UltimatePerformance" --model_name resnet18 --pretrained --use_se --freeze_epochs 10 --warmup_epochs 3 --epochs 60 --batch_size 16 --lr_fc 2e-3 --lr_se 2e-4 --lr_backbone 2e-4 --weight_decay 1e-3

实验9
python train_baseline.py --exp_name "9_ViT_Ultimate" --model_name vit_tiny --pretrained --freeze_epochs 10 --warmup_epochs 3 --epochs 60 --batch_size 16 --lr_fc 2e-3 --lr_backbone 2e-4 --weight_decay 1e-3
