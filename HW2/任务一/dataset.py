import os
import scipy.io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class FlowersDataset(Dataset):
    def __init__(self, root_dir, setid_path, labels_path, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        setid = scipy.io.loadmat(setid_path)
        labels_data = scipy.io.loadmat(labels_path)

        
        if 'labels' in labels_data:
            all_labels = labels_data['labels'].flatten()
        else:
            
            all_labels = None
            for key in labels_data:
                if key.startswith('__'):
                    continue
                arr = labels_data[key]
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    all_labels = arr.flatten()
                    break
        if all_labels is None:
            raise KeyError(
                f"未找到标签字段，请检查 {labels_path} 的内容: {labels_data.keys()}"
            )
        
        all_labels = all_labels.astype(np.int64) - 1

        if mode == 'train':
            indices = setid['trnid'].flatten()
        elif mode == 'val':
            indices = setid['valid'].flatten()
        elif mode == 'test':
            indices = setid['tstid'].flatten()
        else:
            raise ValueError("mode 必须为 'train', 'val' 或 'test'")

        self.image_paths = []
        self.labels = []
        for idx in indices:
          
            filename = f'image_{idx:05d}.jpg'
            path = os.path.join(root_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            self.image_paths.append(path)
            
            self.labels.append(all_labels[idx - 1])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label

# ImageNet 标准化参数
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def get_train_transform(input_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def get_val_test_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

def find_labels_file(data_root):
    """在 data_root 下自动查找标签文件"""
    candidates = ['imagelabels.mat', 'image_labels.mat']
    for fname in candidates:
        full_path = os.path.join(data_root, fname)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"在 {data_root} 中未找到标签文件，尝试过: {candidates}")

if __name__ == '__main__':
    
    IMG_DIR = './102flowers'           
    LABELS_MAT = './imagelabels.mat'   
    SETID_MAT = './setid.mat'          
    train_dataset = FlowersDataset(IMG_DIR, SETID_MAT, LABELS_MAT, mode='train',
                                   transform=get_train_transform())
    val_dataset   = FlowersDataset(IMG_DIR, SETID_MAT, LABELS_MAT, mode='val',
                                   transform=get_val_test_transform())
    test_dataset  = FlowersDataset(IMG_DIR, SETID_MAT, LABELS_MAT, mode='test',
                                   transform=get_val_test_transform())

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    img, label = train_dataset[0]
    print(f"图像张量形状: {img.shape}")
    print(f"标签值示例: {label}")