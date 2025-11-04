"""
Training script for the segmentation model.
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import get_enhanced_unet, DiceFocalLoss, calculate_class_weights
from src.dataset import create_train_val_datasets
from src.metrics import (
    calculate_dice_score,
    calculate_dice_score_for_class,
    calculate_metrics
)
from src.config import TrainingConfig, ModelConfig


def train_model(model, train_loader, val_loader, criterion, config: TrainingConfig, 
                model_config: ModelConfig):
    """
    Train the segmentation model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        config: Training configuration
        model_config: Model configuration
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=config.learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=config.num_epochs
    )
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'{config.log_dir}/unet_{time.strftime("%Y%m%d_%H%M%S")}')
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of epochs: {config.num_epochs}")
    print("-" * 50)
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch loss
            writer.add_scalar('Train/Batch Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_dice_bg = 0.0
        val_dice_benign = 0.0
        val_dice_malignant = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                
                # Calculate Dice scores
                dice_bg = calculate_dice_score_for_class(preds, target, class_id=0)
                dice_benign = calculate_dice_score_for_class(preds, target, class_id=1)
                dice_malignant = calculate_dice_score_for_class(preds, target, class_id=2)
                dice_score = calculate_dice_score(preds, target)
                
                val_dice_bg += dice_bg
                val_dice_benign += dice_benign
                val_dice_malignant += dice_malignant
                val_dice += dice_score
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_dice_bg /= len(val_loader)
        val_dice_benign /= len(val_loader)
        val_dice_malignant /= len(val_loader)
        
        # Log metrics
        writer.add_scalar('Train/Epoch Loss', train_loss, epoch)
        writer.add_scalar('Validation/Epoch Loss', val_loss, epoch)
        writer.add_scalar('Validation/Dice Score', val_dice, epoch)
        writer.add_scalar('Validation/Dice Background', val_dice_bg, epoch)
        writer.add_scalar('Validation/Dice Benign', val_dice_benign, epoch)
        writer.add_scalar('Validation/Dice Malignant', val_dice_malignant, epoch)
        
        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Validation Dice Scores:")
        print(f"    - Background: {val_dice_bg:.5f}")
        print(f"    - Benign: {val_dice_benign:.5f}")
        print(f"    - Malignant: {val_dice_malignant:.5f}")
        print(f"    - Overall: {val_dice:.5f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(config.save_dir, config.model_name)
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print("-" * 50)
    
    writer.close()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(config.save_dir, config.model_name)}")
    
    return model


def evaluate_model(model, val_loader, criterion, device='cuda'):
    """
    Evaluate the model on validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run evaluation on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_precision = 0.0
    val_recall = 0.0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
            preds = torch.argmax(output, dim=1)
            dice_score = calculate_dice_score(preds, target)
            val_dice += dice_score
            
            # Calculate additional metrics
            iou, precision, recall = calculate_metrics(preds, target)
            val_iou += iou
            val_precision += precision
            val_recall += recall
    
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)
    val_precision /= len(val_loader)
    val_recall /= len(val_loader)
    
    print(f"\n{'='*50}")
    print(f"Validation Results:")
    print(f"{'='*50}")
    print(f"Loss:        {val_loss:.4f}")
    print(f"Dice Score:  {val_dice:.4f}")
    print(f"IoU:         {val_iou:.4f}")
    print(f"Precision:   {val_precision:.4f}")
    print(f"Recall:      {val_recall:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Configuration
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Set device
    device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        root_dir=train_config.root_dir,
        val_ratio=train_config.val_ratio,
        image_size=train_config.image_size,
        include_normal=train_config.include_normal,
        stratify=train_config.stratify,
        augment_train=train_config.augment_train
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True, 
        num_workers=train_config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=False, 
        num_workers=train_config.num_workers
    )
    
    # Initialize model
    model = get_enhanced_unet(
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset, device=device)
    print(f"Class Weights: {class_weights}")
    
    # Initialize loss function
    criterion = DiceFocalLoss(
        alpha=train_config.loss_alpha,
        gamma=train_config.loss_gamma,
        class_weights=class_weights
    )
    
    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=train_config,
        model_config=model_config
    )
    
    # Evaluate model
    evaluate_model(trained_model, val_loader, criterion, device=device)

