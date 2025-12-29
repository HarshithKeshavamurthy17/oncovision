"""
Metrics for evaluating segmentation models.
"""
import torch


def calculate_dice_score(preds, targets, smooth=1e-6):
    """
    Calculate Dice score for binary segmentation.
    
    Args:
        preds: Predictions (B, H, W)
        targets: Ground truth (B, H, W)
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    preds = preds.float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def calculate_dice_score_for_class(preds, targets, class_id, smooth=1e-6):
    """
    Calculate Dice score for a specific class.
    
    Args:
        preds: Predictions (B, H, W)
        targets: Ground truth (B, H, W)
        class_id: Class ID to calculate Dice for
        smooth: Smoothing factor
        
    Returns:
        Dice score for the specified class
    """
    pred_class = (preds == class_id).float()
    target_class = (targets == class_id).float()
    intersection = (pred_class * target_class).sum(dim=(1, 2))
    union = pred_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def calculate_metrics(preds, targets, num_classes=3, smooth=1e-6):
    """
    Calculate IoU, Precision, and Recall for all classes.
    
    Args:
        preds: Predictions (B, H, W)
        targets: Ground truth (B, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
        
    Returns:
        Tuple of (IoU, Precision, Recall)
    """
    iou = 0.0
    precision = 0.0
    recall = 0.0

    for class_id in range(num_classes):
        pred_class = (preds == class_id).float()
        target_class = (targets == class_id).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        iou += (intersection + smooth) / (union + smooth)
        precision += (intersection + smooth) / (pred_class.sum() + smooth)
        recall += (intersection + smooth) / (target_class.sum() + smooth)

    iou /= num_classes
    precision /= num_classes
    recall /= num_classes

    return iou, precision, recall




