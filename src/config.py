"""
Configuration file for training and inference parameters.
"""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    encoder_name: str = "resnet50"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 1
    out_channels: int = 3
    decoder_attention_type: str = "scse"
    activation: str = "softmax2d"


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Data
    root_dir: str = "data/train"
    image_size: Tuple[int, int] = (256, 256)
    val_ratio: float = 0.2
    include_normal: bool = True
    stratify: bool = True
    augment_train: bool = True
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss function
    loss_alpha: float = 0.5  # Weight for Dice loss
    loss_gamma: float = 2.0  # Focal loss gamma
    
    # Early stopping
    early_stopping_patience: int = 10
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 0
    
    # Logging
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    model_name: str = "best_model.pth"


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    test_dir: str = "data/test"
    model_path: str = "checkpoints/best_model.pth"
    output_file: str = "submission.csv"
    batch_size: int = 16
    image_size: Tuple[int, int] = (256, 256)
    device: str = "cuda"

