"""
Visualization utilities for model predictions and training data.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import cv2

from src.model import get_enhanced_unet
from src.config import ModelConfig, InferenceConfig


def visualize_predictions(model, val_loader, device, num_samples=5, save_path=None):
    """
    Visualize model predictions on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Optional path to save figures
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            
            # Get first image in batch
            image = data[0].cpu().squeeze().numpy()
            target_mask = target[0].cpu().numpy()
            pred_mask = preds[0].cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title("Input Image", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth
            im1 = axes[1].imshow(target_mask, cmap='jet', vmin=0, vmax=2)
            axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Prediction
            im2 = axes[2].imshow(pred_mask, cmap='jet', vmin=0, vmax=2)
            axes[2].set_title("Prediction", fontsize=12, fontweight='bold')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_sample_{i+1}.png", dpi=150, bbox_inches='tight')
            
            plt.show()


def visualize_dataset_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize samples from the dataset.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        save_path: Optional path to save figures
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        image, mask = dataset[sample_idx]
        
        # Convert to numpy if tensor
        if torch.is_tensor(image):
            image = image.squeeze().cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # Plot image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f"Sample {sample_idx + 1} - Image", fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Plot mask
        im = axes[idx, 1].imshow(mask, cmap='jet', vmin=0, vmax=2)
        axes[idx, 1].set_title(f"Sample {sample_idx + 1} - Mask", fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_dataset_samples.png", dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    import sys
    from src.dataset import create_train_val_datasets
    from src.config import TrainingConfig
    
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        # Visualize dataset samples
        config = TrainingConfig()
        train_dataset, _ = create_train_val_datasets(
            root_dir=config.root_dir,
            val_ratio=config.val_ratio,
            image_size=config.image_size,
            include_normal=config.include_normal,
            stratify=config.stratify,
            augment_train=False
        )
        visualize_dataset_samples(train_dataset, num_samples=5)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "predictions":
        # Visualize model predictions
        model_config = ModelConfig()
        train_config = TrainingConfig()
        inference_config = InferenceConfig()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = get_enhanced_unet(
            encoder_name=model_config.encoder_name,
            encoder_weights=None,
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels
        )
        
        model.load_state_dict(torch.load(inference_config.model_path, map_location=device))
        model = model.to(device)
        
        # Create validation loader
        _, val_dataset = create_train_val_datasets(
            root_dir=train_config.root_dir,
            val_ratio=train_config.val_ratio,
            image_size=train_config.image_size,
            include_normal=train_config.include_normal,
            stratify=train_config.stratify,
            augment_train=False
        )
        
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)
        
        visualize_predictions(model, val_loader, device=device, num_samples=5)
    
    else:
        print("Usage:")
        print("  python -m src.visualize dataset      # Visualize dataset samples")
        print("  python -m src.visualize predictions   # Visualize model predictions")

