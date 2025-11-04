"""
Dataset classes for Breast Ultrasound Image Segmentation.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline(image_size=(256, 256), augment=True):
    """
    Get data augmentation pipeline.
    
    Args:
        image_size: Target image size (height, width)
        augment: Whether to apply augmentations
        
    Returns:
        Albumentations compose object
    """
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussianBlur(blur_limit=(5, 9), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])


class MultiClassBUSIDataset(Dataset):
    """
    Multi-class Breast Ultrasound Image Dataset.
    
    Classes:
        - 0: Background
        - 1: Benign
        - 2: Malignant
    """
    
    def __init__(self, root_dir, image_size=(256, 256), transform=None, samples=None,
                 include_normal=True, augment=False):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing benign, malignant, and normal folders
            image_size: Target image size (height, width)
            transform: Optional transform function
            samples: Pre-defined list of samples (img_path, mask_path, class_id)
            include_normal: Whether to include normal class
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.include_normal = include_normal
        self.augment = augment
        self.class_to_idx = {'background': 0, 'benign': 1, 'malignant': 2}
        
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._scan_dataset()
            
        self.augmentation_pipeline = get_augmentation_pipeline(image_size, augment=augment)

    def _scan_dataset(self):
        """Scan dataset directory and collect all samples."""
        samples = []
        subfolders = ["benign", "malignant", "normal"] if self.include_normal else ["benign", "malignant"]
        
        for sf in subfolders:
            folder_path = os.path.join(self.root_dir, sf)
            mask_paths = glob.glob(os.path.join(folder_path, "*_mask.png"))
            
            for mp in mask_paths:
                img_path = mp.replace("_mask", "")
                if os.path.exists(img_path):
                    class_id = 0 if sf == "normal" else 1 if sf == "benign" else 2
                    samples.append((img_path, mp, class_id))
                    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        img_path, mask_path, class_id = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Create multi-class mask
        binary_mask = (mask > 127).astype(np.uint8)
        multi_class_mask = np.zeros_like(binary_mask) if class_id == 0 else binary_mask * class_id
        
        # Apply augmentation
        if self.augment and self.augmentation_pipeline is not None:
            augmented = self.augmentation_pipeline(image=image, mask=multi_class_mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(multi_class_mask).long()
            
        return image, mask


def create_train_val_datasets(root_dir, val_ratio=0.2, image_size=(256, 256), 
                               include_normal=True, stratify=True, augment_train=True):
    """
    Create train and validation datasets with stratified split.
    
    Args:
        root_dir: Root directory containing class folders
        val_ratio: Validation split ratio
        image_size: Target image size
        include_normal: Whether to include normal class
        stratify: Whether to stratify split by class
        augment_train: Whether to augment training data
        
    Returns:
        train_dataset, val_dataset
    """
    import random
    from collections import defaultdict
    
    subfolders = ["benign", "malignant", "normal"] if include_normal else ["benign", "malignant"]
    all_samples = []
    samples_by_class = defaultdict(list)
    
    # Collect all samples
    for sf in subfolders:
        folder_path = os.path.join(root_dir, sf)
        mask_paths = glob.glob(os.path.join(folder_path, "*_mask.png"))
        
        for mp in mask_paths:
            img_path = mp.replace("_mask", "")
            if os.path.exists(img_path):
                class_id = 0 if sf == "normal" else 1 if sf == "benign" else 2
                sample = (img_path, mp, class_id)
                all_samples.append(sample)
                samples_by_class[class_id].append(sample)
    
    # Split into train and validation
    train_samples = []
    val_samples = []
    
    if stratify:
        # Stratified split by class
        for class_id, samples in samples_by_class.items():
            random.shuffle(samples)
            val_count = int(len(samples) * val_ratio)
            val_samples.extend(samples[:val_count])
            train_samples.extend(samples[val_count:])
    else:
        # Random split
        random.shuffle(all_samples)
        val_count = int(len(all_samples) * val_ratio)
        val_samples = all_samples[:val_count]
        train_samples = all_samples[val_count:]
    
    # Create datasets
    train_dataset = MultiClassBUSIDataset(
        root_dir, 
        image_size=image_size, 
        samples=train_samples, 
        include_normal=include_normal, 
        augment=augment_train
    )
    
    val_dataset = MultiClassBUSIDataset(
        root_dir, 
        image_size=image_size, 
        samples=val_samples, 
        include_normal=include_normal, 
        augment=False
    )
    
    return train_dataset, val_dataset


class BUSITestDataset(Dataset):
    """Dataset class for test images (no masks)."""
    
    def __init__(self, test_dir, image_size=(256, 256)):
        """
        Initialize test dataset.
        
        Args:
            test_dir: Directory containing test images
            image_size: Target image size (height, width)
        """
        self.test_dir = test_dir
        self.image_size = image_size
        self.image_files = [
            f for f in os.listdir(test_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a test image."""
        image_name = self.image_files[idx]
        image_path = os.path.join(self.test_dir, image_name)
        image_id = os.path.splitext(image_name)[0]
        
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_size = original_image.shape
        
        image = cv2.resize(original_image, self.image_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        
        return {
            'image': image_tensor,
            'image_id': image_id,
            'original_h': original_size[0],
            'original_w': original_size[1]
        }

