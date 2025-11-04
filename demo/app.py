"""
Streamlit Demo App for OncoVision
A web-based interface for breast ultrasound image segmentation.
"""
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import get_enhanced_unet
from src.config import ModelConfig

# Page configuration
st.set_page_config(
    page_title="OncoVision - Breast Ultrasound Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = ModelConfig()
    
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
        return model, device
    else:
        return None, device

def preprocess_image(image, image_size=(256, 256)):
    """Preprocess image for model input."""
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

def predict(model, image_tensor, device):
    """Run inference on the image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predictions = torch.argmax(output, dim=1).cpu().numpy()[0]
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    return predictions, probabilities

def create_visualization(original_image, prediction, probabilities):
    """Create visualization of the prediction."""
    # Create colored mask
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    
    # Color mapping: Background=black, Benign=green, Malignant=red
    colored_mask[prediction == 1] = [0, 255, 0]  # Benign - Green
    colored_mask[prediction == 2] = [255, 0, 0]  # Malignant - Red
    
    # Overlay on original image
    overlay = cv2.addWeighted(
        cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        0.6,
        colored_mask,
        0.4,
        0
    )
    
    return colored_mask, overlay

def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">🔬 OncoVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-based Breast Ultrasound Image Segmentation</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        **OncoVision** is an AI-powered tool for automated segmentation of breast ultrasound images.
        
        ### Features:
        - Multi-class segmentation (Background, Benign, Malignant)
        - Real-time predictions
        - Visual overlay of results
        
        ### Model:
        - Architecture: U-Net with ResNet50 encoder
        - Training: Combined Dice-Focal Loss
        - Input: Grayscale ultrasound images
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This tool is for educational and research purposes only. 
        Not intended for clinical diagnosis.
        """)
        
        st.header("🔗 Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/HarshithKeshavamurthy17/oncovision)
        - [Documentation](https://github.com/HarshithKeshavamurthy17/oncovision#readme)
        """)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please ensure 'checkpoints/best_model.pth' exists.")
        st.info("To train the model, run: `python -m src.train`")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a breast ultrasound image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color breast ultrasound image"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process and predict
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    # Preprocess
                    image_tensor, processed_image = preprocess_image(image_array)
                    
                    # Predict
                    prediction, probabilities = predict(model, image_tensor, device)
                    
                    # Visualize
                    colored_mask, overlay = create_visualization(
                        processed_image, prediction, probabilities
                    )
                    
                    # Store in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['probabilities'] = probabilities
                    st.session_state['overlay'] = overlay
                    st.session_state['mask'] = colored_mask
                    st.session_state['original'] = processed_image
                    st.rerun()
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            # Display overlay
            st.image(st.session_state['overlay'], caption="Segmentation Overlay", use_container_width=True)
            
            # Display mask
            st.image(st.session_state['mask'], caption="Segmentation Mask", use_container_width=True)
            
            # Statistics
            st.subheader("📈 Statistics")
            prediction = st.session_state['prediction']
            probabilities = st.session_state['probabilities']
            
            total_pixels = prediction.size
            benign_pixels = np.sum(prediction == 1)
            malignant_pixels = np.sum(prediction == 2)
            background_pixels = np.sum(prediction == 0)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Background", f"{background_pixels/total_pixels*100:.1f}%")
            with col_b:
                st.metric("Benign", f"{benign_pixels/total_pixels*100:.1f}%", 
                         delta="Normal tissue" if benign_pixels > 0 else None)
            with col_c:
                st.metric("Malignant", f"{malignant_pixels/total_pixels*100:.1f}%",
                         delta="⚠️" if malignant_pixels > 0 else None,
                         delta_color="inverse" if malignant_pixels > 0 else "normal")
            
            # Probability distribution
            st.subheader("🎯 Class Probabilities")
            prob_dict = {
                "Background": float(probabilities[0].mean()),
                "Benign": float(probabilities[1].mean()),
                "Malignant": float(probabilities[2].mean())
            }
            st.bar_chart(prob_dict)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using PyTorch, Streamlit, and Segmentation Models</p>
        <p>© 2024 OncoVision Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

