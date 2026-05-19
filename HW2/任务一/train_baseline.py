import os
import copy
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


import torchvision
from packaging import version
if version.parse(torchvision.__version__) < version.parse("0.13.0"):
    raise RuntimeError(
        f"当前 torchvision 版本为 {torchvision.__version__}，请升级至 >=0.13.0 "
        "以使用 ResNet18_Weights 等新式预训练接口。"
    )


import timm
from dataset import FlowersDataset, get_train_transform, get_val_test_transform, find_labels_file
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import swanlab
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ================== 1. SE 模块及 SE-BasicBlock ==================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, init_scale=4.0):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, init_scale)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError('SEBasicBlock 仅支持 groups=1 和 base_width=64')
        if dilation > 1: raise NotImplementedError("dilation > 1 暂不支持")
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ================== 2. SE-ResNet 模型 ==================
class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=102, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, reduction=16):
        super(SEResNet, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1
        self.reduction = reduction

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, reduction=self.reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, reduction=self.reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

def se_resnet18(num_classes=102, **kwargs): return SEResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
def se_resnet34(num_classes=102, **kwargs): return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

# ================== 3. 统一的模型构建函数 ==================
def build_model(model_name='resnet18', num_classes=102, use_pretrained=True, use_se=False, reduction=16):
    if model_name in ['vit_tiny', 'swin_tiny']:
        timm_model_name = 'vit_tiny_patch16_224' if model_name == 'vit_tiny' else 'swin_tiny_patch4_window7_224'
        print(f"正在通过 timm 构建 {timm_model_name}，预训练: {use_pretrained}")

        # ===== 本地权重路径 =====
        local_checkpoint = {
            'vit_tiny': './pretrained/vit_tiny_patch16_224/model.safetensors',
        }.get(model_name, None)

        if use_pretrained and local_checkpoint and os.path.exists(local_checkpoint):
            print(f"从本地加载预训练权重: {local_checkpoint}")
            model = timm.create_model(
                timm_model_name,
                pretrained=False,
                num_classes=1000,              
                checkpoint_path=local_checkpoint, 
            )
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            print(f"分类头已替换: 1000 -> {num_classes} (in_features={in_features})")
            return model
        
        # 如果本地路径不存在，回退到原来的在线下载逻辑
        print("本地权重路径不存在，尝试从 HuggingFace 在线下载...")
        weights = 'default' if use_pretrained else None
        model = timm.create_model(timm_model_name, pretrained=weights, num_classes=num_classes)
        return model

    if use_se:
        if model_name == 'resnet18':
            model = se_resnet18(num_classes=num_classes, reduction=reduction)
        elif model_name == 'resnet34':
            model = se_resnet34(num_classes=num_classes, reduction=reduction)
        else:
            raise ValueError("SE 版本当前仅支持 resnet18 或 resnet34")

        if use_pretrained:
            print("正在加载官方预训练权重（部分加载，SE 模块将随机初始化）...")
            if model_name == 'resnet18':
                pretrained_dict = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
            elif model_name == 'resnet34':
                pretrained_dict = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"成功加载 {len(pretrained_dict)} 个预训练参数")
        return model
    else:
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
            model = models.resnet34(weights=weights)
        else:
            raise ValueError("仅支持 resnet18, resnet34, vit_tiny, swin_tiny")
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model


# ================== 4. 分层学习率优化器 ==================
def get_optimizer(model, lr_fc=1e-3, lr_se=1e-4, lr_backbone=1e-4, weight_decay=1e-4):
    head_params = []; se_params = []; backbone_params = []
    head_keywords = ['fc', 'head']
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        
        is_head = name.split('.')[0] in head_keywords
        is_se = 'se' in name
        
        if is_se: 
            se_params.append(param)
        elif is_head: 
            head_params.append(param)
        else: 
            backbone_params.append(param)

    param_groups = [
        {'params': head_params, 'lr': lr_fc},
        {'params': backbone_params, 'lr': lr_backbone},
    ]
    if se_params: param_groups.insert(1, {'params': se_params, 'lr': lr_se})
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
    return optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)

# ================== 5. 自定义学习率调度 (Freeze + Warmup + Cosine) ==================
def adjust_learning_rate(optimizer, epoch, config):
    """手动控制学习率: 支持 Freeze阶段 -> Warmup阶段 -> Cosine Decay阶段"""
    freeze_epochs = config.get('freeze_epochs', 0)
    pretrained = config.get('pretrained', True) # 获取是否预训练的标志
    warmup_epochs = config.get('warmup_epochs', 0)
    total_epochs = config.get('epochs', 30)

    # 只有在确实使用了预训练且处于冻结阶段时，才跳过学习率调整
    if epoch <= freeze_epochs and pretrained:
        return

    # 2. 计算解冻后的相对 epoch
    relative_epoch = epoch - freeze_epochs if pretrained else epoch
    relative_total = total_epochs - freeze_epochs if pretrained else total_epochs

    # 3. Warmup 阶段
    if warmup_epochs > 0 and relative_epoch <= warmup_epochs:
        warmup_factor = relative_epoch / warmup_epochs
        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg: pg['initial_lr'] = pg['lr']
            pg['lr'] = pg['initial_lr'] * warmup_factor
            
    # 4. Cosine Decay 阶段
    else:
        cosine_relative_epoch = relative_epoch - warmup_epochs
        cosine_total_epochs = relative_total - warmup_epochs
        
        if cosine_total_epochs <= 0: return # 防止除0
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_relative_epoch / cosine_total_epochs))
        for pg in optimizer.param_groups:
            if 'initial_lr' not in pg: pg['initial_lr'] = pg['lr']
            pg['lr'] = pg['initial_lr'] * cosine_factor

# ================== 6. 训练、评估、绘图 ==================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0; running_corrects = 0; total = 0
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{running_corrects/total:.4f}'})
    return running_loss / total, running_corrects / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0; running_corrects = 0; total = 0
    all_probs = []; all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device); labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    classes = np.arange(102)
    all_labels_one_hot = label_binarize(all_labels, classes=classes)
    mAP = average_precision_score(all_labels_one_hot, all_probs, average='macro')
    return running_loss / total, running_corrects / total, mAP

def plot_history(history, save_path='training_curves.png'):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_mAP'], 'g-', label='Val mAP')
    plt.xlabel('Epochs'); plt.ylabel('mAP'); plt.legend(); plt.title('Validation mAP')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Training curves saved to {save_path}')

# ================== 7. 核心训练接口 ==================
def train_and_evaluate(config, datasets, device, save_dir='.', plot_curve=False):
    model_name = config.get('model_name', 'resnet18')
    epochs = config.get('epochs', 30)
    lr_fc = config.get('lr_fc', 1e-3)
    lr_se = config.get('lr_se', 1e-3)
    lr_backbone = config.get('lr_backbone', 1e-4)
    batch_size = config.get('batch_size', 32)
    weight_decay = config.get('weight_decay', 1e-4)
    pretrained = config.get('pretrained', True)
    use_se = config.get('use_se', False)
    reduction = config.get('reduction', 16)
    input_size = config.get('input_size', 224)
    num_workers = config.get('num_workers', 4)
    save_name = config.get('save_name', 'best_model.pth')
    exp_name = config.get('exp_name', 'flower102_exp')
    use_swanlab = config.get('use_swanlab', True)
    freeze_epochs = config.get('freeze_epochs', 0)
    warmup_epochs = config.get('warmup_epochs', 0) 

    if use_swanlab:
        swanlab.init(project="flower102", experiment_name=exp_name, config=config)

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    model = build_model(model_name, num_classes=102, use_pretrained=pretrained, use_se=use_se, reduction=reduction)
    model = model.to(device)

    # 冻结逻辑
    if freeze_epochs > 0 and pretrained:
        print(f"冻结 Backbone 前 {freeze_epochs} 个 Epoch，仅训练 Head/FC 层...")
        for name, param in model.named_parameters():
            
            if name.split('.')[0] not in ['fc', 'head']:
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr_fc=lr_fc, lr_se=lr_se, lr_backbone=lr_backbone, weight_decay=weight_decay)
    
    # 记录初始学习率，为后续自定义衰减做准备
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_mAP': []}
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        # 解冻逻辑
        if epoch == freeze_epochs + 1 and freeze_epochs > 0 and pretrained:
            print("解冻 Backbone，开始联合微调...")
            for name, param in model.named_parameters():
                param.requires_grad = True
            
            optimizer = get_optimizer(model, lr_fc=lr_fc, lr_se=lr_se, lr_backbone=lr_backbone, weight_decay=weight_decay)
            for pg in optimizer.param_groups:
                pg['initial_lr'] = pg['lr']

        # 应用自定义学习率调整 
        adjust_learning_rate(optimizer, epoch, config)

        print(f'\nEpoch {epoch}/{epochs}')
        print('-' * 30)
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc, val_mAP = evaluate(model, dataloaders['val'], criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mAP'].append(val_mAP)

        lr_dict = {}
        for i, pg in enumerate(optimizer.param_groups):
            lr_dict[f"lr_group_{i}"] = pg['lr']

        if use_swanlab:
            swanlab.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_mAP": val_mAP,
                "epoch": epoch, **lr_dict
            })

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} mAP: {val_mAP:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs(save_dir, exist_ok=True)
            torch.save(best_model_wts, os.path.join(save_dir, save_name))
            print(f'Best model saved (val acc: {best_acc:.4f})')

    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, test_mAP = evaluate(model, dataloaders['test'], criterion, device)
    print(f'Final Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} mAP: {test_mAP:.4f}')
    
    if use_swanlab:
        swanlab.log({"test_loss": test_loss, "test_acc": test_acc, "test_mAP": test_mAP})
        swanlab.finish()

    if plot_curve:
        plot_history(history, save_path=os.path.join(save_dir, save_name.replace('.pth', '_curves.png')))

    return best_acc, best_model_wts, history

# ================== 8. 命令行入口 ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune ResNet/ViT on 102 Flowers')
    parser.add_argument('--data_root', type=str, default='./', help='数据集根目录')
    parser.add_argument('--img_dir', type=str, default='102flowers', help='图片子文件夹')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vit_tiny', 'swin_tiny'], help='选择模型架构')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='使用ImageNet预训练')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='不使用预训练')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--use_se', action='store_true', default=False, help='使用SE注意力 (仅对ResNet有效)')
    parser.add_argument('--reduction', type=int, default=16, help='SE模块压缩比')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--freeze_epochs', type=int, default=5, help='冻结Backbone的Epoch数，0表示不冻结')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='解冻Backbone后的Warmup轮数，0表示不进行Warmup')
    parser.add_argument('--lr_fc', type=float, default=1e-3, help='分类头学习率')
    parser.add_argument('--lr_se', type=float, default=1e-4, help='SE模块学习率')
    parser.add_argument('--lr_backbone', type=float, default=1e-4, help='主干网络学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default=None, help='模型保存路径，若不指定则自动保存至 checkpoints/<exp_name>/best_model.pth')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--exp_name', type=str, default='flower102_baseline', help='SwanLab实验名称')
    parser.add_argument('--no_swanlab', action='store_true', default=False, help='禁用SwanLab记录')
    args = parser.parse_args()

    set_seed(args.seed)

    if args.save_path is None:
        args.save_path = os.path.join('checkpoints', args.exp_name, 'best_model.pth')

    labels_mat = find_labels_file(args.data_root)
    setid_mat = os.path.join(args.data_root, 'setid.mat')
    img_dir = os.path.join(args.data_root, args.img_dir)

    train_dataset = FlowersDataset(img_dir, setid_mat, labels_mat, mode='train', transform=get_train_transform(args.input_size))
    val_dataset = FlowersDataset(img_dir, setid_mat, labels_mat, mode='val', transform=get_val_test_transform(args.input_size))
    test_dataset = FlowersDataset(img_dir, setid_mat, labels_mat, mode='test', transform=get_val_test_transform(args.input_size))
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    config = {
        'model_name': args.model_name, 'pretrained': args.pretrained,
        'use_se': args.use_se, 'reduction': args.reduction,
        'epochs': args.epochs, 'freeze_epochs': args.freeze_epochs,
        'warmup_epochs': args.warmup_epochs,
        'lr_fc': args.lr_fc, 'lr_se': args.lr_se, 'lr_backbone': args.lr_backbone,
        'batch_size': args.batch_size, 'weight_decay': args.weight_decay,
        'num_workers': args.num_workers, 'input_size': args.input_size,
        'save_name': os.path.basename(args.save_path),
        'exp_name': args.exp_name, 'use_swanlab': not args.no_swanlab,
    }

    train_and_evaluate(config, datasets, device, save_dir=os.path.dirname(args.save_path) or '.', plot_curve=True)