# dataset.py - 修复版
"""
Stanford Background Dataset 数据加载
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StanfordBackgroundDataset(Dataset):
    """
    Stanford Background Dataset (SBD)
    """
    def __init__(self, root_dir, transform=None, target_transform=None, 
                 binary=True, label_type='regions'):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.binary = binary
        self.label_type = label_type
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"标签目录不存在: {self.labels_dir}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = sorted([f for f in os.listdir(self.images_dir) 
                             if f.lower().endswith(image_extensions)])
        
        # 获取对应的标签文件（.txt格式）
        self.labels = []
        for img_file in self.images:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.{label_type}.txt"
            label_path = os.path.join(self.labels_dir, label_file)
            if os.path.exists(label_path):
                self.labels.append(label_file)
        
        # 只保留有对应标签的图像
        self.images = [self.images[i] for i in range(len(self.labels))]
        
        print(f"数据集加载完成: {len(self.images)} 对图像-标签")
        print(f"标签类型: {label_type}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        mask = self._load_label_txt(label_path, image.size)
        
        image_np = np.array(image)
        
        if self.binary:
            mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_np = self.transform(image_np)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        # 修复：numpy数组使用 torch 转换和添加通道维度
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        
        # 确保有通道维度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # 现在 mask 是 tensor，可以调用 unsqueeze
        
        return image_np, mask
    
    def _load_label_txt(self, txt_path, image_size):
        """从.txt文件加载标签"""
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        label_rows = []
        for line in lines:
            line = line.strip()
            if line:
                row_values = [int(x) for x in line.split()]
                label_rows.append(row_values)
        
        mask = np.array(label_rows, dtype=np.int64)
        
        # 确保标签尺寸与图像匹配
        if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize(image_size, Image.NEAREST)
            mask = np.array(mask_img)
        
        return mask


def get_transforms(image_size=(256, 256)):
    """获取数据预处理变换"""
    
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    return img_transform, mask_transform


def check_dataset_structure(root_dir):
    """检查数据集结构"""
    print(f"\n检查数据集结构: {root_dir}")
    print("="*50)
    
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')
    
    print(f"图像目录存在: {os.path.exists(images_dir)}")
    print(f"标签目录存在: {os.path.exists(labels_dir)}")
    
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        print(f"图像文件数量: {len(images)}")
        if len(images) > 0:
            print(f"图像示例: {images[:3]}")
    
    if os.path.exists(labels_dir):
        all_labels = os.listdir(labels_dir)
        layers = [f for f in all_labels if f.endswith('.layers.txt')]
        regions = [f for f in all_labels if f.endswith('.regions.txt')]
        surfaces = [f for f in all_labels if f.endswith('.surfaces.txt')]
        
        print(f"标签文件统计:")
        print(f"  .layers.txt: {len(layers)} 个")
        print(f"  .regions.txt: {len(regions)} 个")
        print(f"  .surfaces.txt: {len(surfaces)} 个")
        
        if len(layers) > 0:
            print(f"标签示例: {layers[:3]}")
    
    return True