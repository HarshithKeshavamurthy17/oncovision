"""
Gradio Demo App for OncoVision
Alternative web-based interface using Gradio.
"""
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import get_enhanced_unet
from src.config import ModelConfig

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_config = ModelConfig()

model = None

def load_model():
    """Load the trained model."""
    global model
    model = get_enhanced_unet(
        encoder_name=model_config.encoder_name,
        encoder_weights=None,
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels
    )
    
    model_path = "checkpoints/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    return model is not None

def preprocess_image(image, image_size=(256, 256)):
    """Preprocess image for model input."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    
    return image_tensor, image

def predict(image):
    """Run inference on the image."""
    if model is None:
        return None, "Model not loaded. Please train the model first."
    
    # Preprocess
    image_tensor, processed_image = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predictions = torch.argmax(output, dim=1).cpu().numpy()[0]
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    # Create visualization
    colored_mask = np.zeros((*predictions.shape, 3), dtype=np.uint8)
    colored_mask[predictions == 1] = [0, 255, 0]  # Benign - Green
    colored_mask[predictions == 2] = [255, 0, 0]  # Malignant - Red
    
    # Overlay
    overlay = cv2.addWeighted(
        cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        0.6,
        colored_mask,
        0.4,
        0
    )
    
    # Calculate statistics
    total_pixels = predictions.size
    benign_pixels = np.sum(predictions == 1)
    malignant_pixels = np.sum(predictions == 2)
    background_pixels = np.sum(predictions == 0)
    
    stats = f"""
    **Segmentation Statistics:**
    - Background: {background_pixels/total_pixels*100:.1f}%
    - Benign: {benign_pixels/total_pixels*100:.1f}%
    - Malignant: {malignant_pixels/total_pixels*100:.1f}%
    
    **Class Probabilities:**
    - Background: {probabilities[0].mean():.3f}
    - Benign: {probabilities[1].mean():.3f}
    - Malignant: {probabilities[2].mean():.3f}
    """
    
    return overlay, stats

# Create Gradio interface
if load_model():
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Breast Ultrasound Image"),
        outputs=[
            gr.Image(type="numpy", label="Segmentation Result"),
            gr.Markdown(label="Statistics")
        ],
        title="🔬 OncoVision - Breast Ultrasound Segmentation",
        description="""
        Upload a breast ultrasound image to get automated segmentation results.
        The model identifies benign tumors (green), malignant tumors (red), and background tissue.
        
        **Note:** This tool is for educational purposes only, not for clinical diagnosis.
        """,
        examples=[
            "data/test/image_000.png",
            "data/test/image_001.png",
            "data/test/image_002.png"
        ] if os.path.exists("data/test") else None,
        article="""
        <div style='text-align: center;'>
            <p>Built with PyTorch and Gradio</p>
            <p><a href='https://github.com/HarshithKeshavamurthy17/oncovision'>GitHub Repository</a></p>
        </div>
        """
    )
    
    if __name__ == "__main__":
        demo.launch(share=True)  # Set share=True for public URL
else:
    print("Error: Model not found. Please train the model first.")

