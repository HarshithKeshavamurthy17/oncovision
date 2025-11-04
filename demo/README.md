# OncoVision Demo

Interactive web-based demo for the OncoVision breast ultrasound segmentation project.

## Quick Start

### Streamlit Demo

```bash
# Install dependencies
pip install streamlit

# Run the demo
streamlit run demo/app.py
```

The demo will open in your browser at `http://localhost:8501`

### Gradio Demo

```bash
# Install dependencies
pip install gradio

# Run the demo
python demo/gradio_app.py
```

The demo will open in your browser and provide a public shareable link.

## Features

- **Image Upload**: Upload breast ultrasound images
- **Real-time Segmentation**: Get instant predictions
- **Visual Overlay**: See segmentation results overlaid on original image
- **Statistics**: View class probabilities and pixel distribution
- **Color-coded Results**: 
  - Green = Benign tumors
  - Red = Malignant tumors
  - Black = Background

## Requirements

- Trained model at `checkpoints/best_model.pth`
- Python 3.8+
- PyTorch
- Streamlit or Gradio

## Deployment

See [DEPLOYMENT.md](../DEPLOYMENT.md) for detailed deployment instructions.

## Usage

1. Upload an image using the file uploader
2. Click "Analyze Image"
3. View segmentation results, statistics, and probabilities

## Notes

- The demo requires a trained model to function
- For best results, use grayscale ultrasound images
- This tool is for educational purposes only

