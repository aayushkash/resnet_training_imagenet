import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random
from typing import Optional

class ImageNetDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None, debug: bool = False, debug_fraction: float = 0.05):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all class folders
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = root_dir / class_name
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.JPEG', '.jpg', '.png'))]
            
            if debug:
                # Take only a fraction of images in debug mode
                num_samples = max(1, int(len(image_files) * debug_fraction))
                image_files = random.sample(image_files, num_samples)
            
            for img_file in image_files:
                self.image_paths.append(class_dir / img_file)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(config):
    train_dataset = ImageNetDataset(
        config.TRAIN_DIR,
        transform=get_transforms(train=True),
        debug=config.DEBUG,
        debug_fraction=config.DEBUG_FRACTION
    )
    
    val_dataset = ImageNetDataset(
        config.VAL_DIR,
        transform=get_transforms(train=False),
        debug=config.DEBUG,
        debug_fraction=config.DEBUG_FRACTION
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader 