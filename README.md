# OncoVision - Breast Ultrasound Image Segmentation

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Deep Learning-based Multi-class Segmentation for Breast Ultrasound Images**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Deployment](#-deployment) • [Project Structure](#project-structure) • [Results](#results)

</div>

## 📋 Overview

OncoVision is a deep learning project for automated segmentation of breast ultrasound images. The system uses a U-Net architecture with a ResNet50 encoder to perform multi-class segmentation, distinguishing between benign tumors, malignant tumors, and background/normal tissue.

This project is designed for medical image analysis and can be used as a tool to assist in breast cancer detection and diagnosis.

## ✨ Features

- **Multi-class Segmentation**: Classifies breast ultrasound images into background, benign, and malignant regions
- **Advanced Architecture**: U-Net with ResNet50 encoder pre-trained on ImageNet
- **Robust Training**: Combined Dice-Focal Loss with class weights to handle imbalanced datasets
- **Data Augmentation**: Extensive augmentation pipeline including geometric and photometric transformations
- **Comprehensive Metrics**: Dice Score, IoU, Precision, and Recall for each class
- **Easy to Use**: Modular codebase with configuration-based training and inference
- **Production Ready**: Clean code structure suitable for portfolio and deployment
- **Interactive Demo**: Web-based Streamlit and Gradio interfaces for real-time predictions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/oncovision.git
   cd oncovision
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
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

## 📖 Usage

### 🚀 Live Demo

**🌐 Try the live demo:** [https://oncovision.up.railway.app](https://oncovision.up.railway.app)

💡 **Note:** This app uses Railway's free tier. If it hasn't been visited recently, it may take 10-30 seconds to wake up. This is normal behavior for free hosting.

**Local Demo:**

```bash
# Streamlit Demo
streamlit run demo/app.py

# Or Gradio Demo
python demo/gradio_app.py
```

### 🚢 Deployment

This project is deployed on Railway.app. For detailed deployment instructions, see:
- **[RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)** - Complete Railway deployment guide
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Alternative deployment options (Streamlit Cloud, Heroku, etc.)
- **[DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)** - Deployment checklist
- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Common issues and solutions

### Training

Train the model with default configuration:

```bash
python -m src.train
```

Or customize training parameters in `src/config.py`:

```python
train_config = TrainingConfig(
    root_dir="data/train",
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-4,
    image_size=(256, 256)
)
```

### Inference

Generate predictions on test images:

```bash
python -m src.inference
```

The script will:
- Load the trained model from `checkpoints/best_model.pth`
- Process all images in `data/test/`
- Generate a submission file `submission.csv` with RLE-encoded predictions

### Visualization

View training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

## 📁 Project Structure

```
oncovision/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration classes
│   ├── dataset.py           # Dataset classes and data loaders
│   ├── model.py             # Model definitions and loss functions
│   ├── metrics.py           # Evaluation metrics
│   ├── train.py             # Training script
│   └── inference.py         # Inference script
├── data/
│   ├── train/               # Training data (benign, malignant, normal)
│   └── test/                 # Test images
├── checkpoints/              # Saved model weights
├── runs/                     # TensorBoard logs
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

## 🏗️ Architecture

### Model Architecture

- **Encoder**: ResNet50 (pre-trained on ImageNet)
- **Decoder**: U-Net with SCSE (Spatial and Channel Squeeze & Excitation) attention
- **Output**: 3-channel softmax (Background, Benign, Malignant)

### Loss Function

Combined Dice-Focal Loss:
- **Dice Loss**: Handles class imbalance effectively
- **Focal Loss**: Focuses on hard examples
- **Class Weights**: Automatically calculated from dataset distribution

### Data Augmentation

- Horizontal/Vertical flips
- Rotation (up to 30°)
- Shift, scale, and rotate
- Elastic transforms
- Grid distortion
- Optical distortion
- Brightness/contrast adjustments
- Gaussian blur
- Gamma correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

## 📊 Results

Example validation metrics:
- **Dice Score**: ~0.56
- **IoU**: Calculated per class
- **Precision/Recall**: Balanced across classes

## 🔧 Configuration

All configuration parameters are defined in `src/config.py`:

### ModelConfig
- `encoder_name`: Encoder architecture (default: "resnet50")
- `encoder_weights`: Pre-trained weights (default: "imagenet")
- `in_channels`: Input channels (default: 1 for grayscale)
- `out_channels`: Number of classes (default: 3)

### TrainingConfig
- `batch_size`: Batch size (default: 16)
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate (default: 1e-4)
- `val_ratio`: Validation split ratio (default: 0.2)
- `early_stopping_patience`: Early stopping patience (default: 10)

## 🧪 Experiments

The project includes:
- **Stratified train/validation split**: Ensures balanced class distribution
- **OneCycleLR scheduler**: Optimizes learning rate during training
- **Gradient clipping**: Prevents gradient explosion
- **TensorBoard logging**: Real-time training monitoring

## 📝 Dataset Format

### Training Data
- Images: PNG format, grayscale
- Masks: PNG format with `_mask.png` suffix
- Naming: `{class_name} ({number}).png` and `{class_name} ({number})_mask.png`

### Test Data
- Images: PNG format, grayscale
- Naming: `image_XXX.png` (where XXX is the image ID)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: BUSI (Breast Ultrasound Images) Dataset
- **Libraries**: PyTorch, Segmentation Models PyTorch, Albumentations
- **Architecture**: U-Net by Ronneberger et al.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🎯 Future Improvements

- [x] Web interface for inference (Streamlit & Gradio)
- [x] Railway deployment
- [ ] Support for additional encoder architectures
- [ ] Test-time augmentation
- [ ] Model ensemble capabilities
- [ ] Docker containerization
- [ ] API endpoint for real-time predictions

---

<div align="center">
Made with ❤️ for medical image analysis
</div>

