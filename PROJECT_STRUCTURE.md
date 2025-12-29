# Project Structure

This document describes the organization of the OncoVision project.

## Directory Structure

```
oncovision/
│
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration classes
│   ├── dataset.py                # Dataset classes and data loaders
│   ├── model.py                 # Model definitions and loss functions
│   ├── metrics.py               # Evaluation metrics
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference script
│   └── visualize.py             # Visualization utilities
│
├── data/                         # Data directory
│   ├── train/                   # Training data
│   │   ├── benign/              # Benign tumor images and masks
│   │   ├── malignant/           # Malignant tumor images and masks
│   │   └── normal/              # Normal tissue images and masks
│   └── test/                    # Test images (no masks)
│
├── scripts/                      # Utility scripts
│   ├── train.sh                 # Training script
│   └── inference.sh             # Inference script
│
├── checkpoints/                  # Saved model weights (created during training)
├── runs/                         # TensorBoard logs (created during training)
│
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow
│
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # Main documentation
├── QUICKSTART.md                # Quick start guide
├── CONTRIBUTING.md              # Contribution guidelines
├── PROJECT_STRUCTURE.md         # This file
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup script

```

## Module Descriptions

### `src/config.py`
- `ModelConfig`: Model architecture configuration
- `TrainingConfig`: Training hyperparameters and settings
- `InferenceConfig`: Inference parameters

### `src/dataset.py`
- `MultiClassBUSIDataset`: Main dataset class for training/validation
- `BUSITestDataset`: Dataset class for test images
- `create_train_val_datasets()`: Creates train/val splits
- `get_augmentation_pipeline()`: Data augmentation pipeline

### `src/model.py`
- `get_enhanced_unet()`: Creates U-Net model with encoder
- `DiceFocalLoss`: Combined loss function
- `calculate_class_weights()`: Computes class weights from dataset

### `src/metrics.py`
- `calculate_dice_score()`: Dice score calculation
- `calculate_dice_score_for_class()`: Per-class Dice score
- `calculate_metrics()`: IoU, Precision, Recall

### `src/train.py`
- `train_model()`: Main training loop
- `evaluate_model()`: Model evaluation

### `src/inference.py`
- `generate_submission()`: Generate predictions on test set
- `rle_encode_mask()`: Run-length encoding for masks
- `combined_encode()`: Multi-class mask encoding

### `src/visualize.py`
- `visualize_predictions()`: Visualize model predictions
- `visualize_dataset_samples()`: Visualize dataset samples

## File Naming Conventions

- **Training images**: `{class_name} ({number}).png`
- **Training masks**: `{class_name} ({number})_mask.png`
- **Test images**: `image_XXX.png`
- **Model checkpoints**: `best_model.pth` (saved in `checkpoints/`)

## Data Flow

1. **Training**:
   - Data loaded from `data/train/`
   - Model trained and saved to `checkpoints/`
   - Logs saved to `runs/`

2. **Inference**:
   - Model loaded from `checkpoints/best_model.pth`
   - Test images loaded from `data/test/`
   - Predictions saved to `submission.csv`

## Configuration

All configuration is centralized in `src/config.py`:
- Model architecture settings
- Training hyperparameters
- Data paths and preprocessing
- Inference settings

## Extending the Project

To add new features:
1. Add new modules to `src/`
2. Update `src/config.py` if new parameters needed
3. Update `requirements.txt` if new dependencies needed
4. Update documentation




