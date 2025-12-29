# Quick Start Guide

This guide will help you get started with OncoVision quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/oncovision.git
cd oncovision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Ensure your data is organized as follows:

```
data/
├── train/
│   ├── benign/
│   │   ├── benign (1).png
│   │   ├── benign (1)_mask.png
│   │   └── ...
│   ├── malignant/
│   │   ├── malignant (1).png
│   │   ├── malignant (1)_mask.png
│   │   └── ...
│   └── normal/
│       ├── normal (1).png
│       ├── normal (1)_mask.png
│       └── ...
└── test/
    ├── image_000.png
    ├── image_001.png
    └── ...
```

## Training

### Basic Training

```bash
python -m src.train
```

### Custom Configuration

Edit `src/config.py` to customize training parameters:

```python
train_config = TrainingConfig(
    root_dir="data/train",
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-4,
    image_size=(256, 256),
    val_ratio=0.2
)
```

## Inference

### Generate Predictions

```bash
python -m src.inference
```

This will:
1. Load the trained model from `checkpoints/best_model.pth`
2. Process images from `data/test/`
3. Generate `submission.csv` with predictions

### Custom Inference Paths

Edit `src/config.py`:

```python
inference_config = InferenceConfig(
    test_dir="data/test",
    model_path="checkpoints/best_model.pth",
    output_file="predictions.csv"
)
```

## Visualization

### View Dataset Samples

```bash
python -m src.visualize dataset
```

### View Model Predictions

```bash
python -m src.visualize predictions
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs
```

Then open `http://localhost:6006` in your browser.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in `src/config.py`
   - Use smaller `image_size`

2. **Model Not Found**
   - Ensure you've trained the model first
   - Check `model_path` in `InferenceConfig`

3. **Import Errors**
   - Ensure you're in the project root directory
   - Activate your virtual environment
   - Install all requirements: `pip install -r requirements.txt`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the code in `src/` directory
- Customize configurations for your use case
- Experiment with different architectures and hyperparameters




