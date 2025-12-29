"""
Model definitions for Breast Ultrasound Image Segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional


def get_enhanced_unet(encoder_name='resnet50', encoder_weights='imagenet', 
                      in_channels=1, out_channels=3):
    """
    Create an enhanced U-Net model with pre-trained encoder.
    
    Args:
        encoder_name: Name of the encoder (e.g., 'resnet50', 'efficientnet-b0')
        encoder_weights: Pre-trained weights ('imagenet' or None)
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output classes (3 for background, benign, malignant)
        
    Returns:
        U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        decoder_attention_type='scse',
        activation='softmax2d',
    )
    return model


class DiceFocalLoss(nn.Module):
    """
    Combined Dice and Focal Loss for multi-class segmentation.
    
    This loss combines:
    - Dice Loss: Good for imbalanced datasets
    - Focal Loss: Focuses on hard examples
    """
    
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1e-6, class_weights=None):
        """
        Initialize loss function.
        
        Args:
            alpha: Weight for Dice loss (1-alpha for Focal loss)
            gamma: Focal loss gamma parameter
            smooth: Smoothing factor for numerical stability
            class_weights: Class weights tensor for handling class imbalance
        """
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Model predictions (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            Combined loss value
        """
        targets = targets.long()
        inputs_softmax = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Dice Loss
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(2, 3))
        union = inputs_softmax.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Focal Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Combined loss
        loss = self.alpha * dice_loss.mean() + (1 - self.alpha) * focal_loss.mean()
        
        return loss


def calculate_class_weights(dataset, num_classes=3, device='cuda'):
    """
    Calculate class weights based on dataset distribution.
    
    Args:
        dataset: Dataset to calculate weights from
        num_classes: Number of classes
        device: Device to place weights on
        
    Returns:
        Class weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for _, mask in dataset:
        class_counts += torch.bincount(mask.flatten(), minlength=num_classes)
    
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights /= class_weights.sum()
    
    return class_weights.to(device)




