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

def download_model_from_url(url, save_path):
    """Download model from URL if it doesn't exist."""
    import urllib.request
    import ssl
    
    if os.path.exists(save_path):
        return True
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Create SSL context to handle HTTPS
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        st.warning(f"Could not download model from URL: {e}")
        return False

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
    
    # Try multiple possible paths for the model
    model_paths = [
        "checkpoints/best_model.pth",
        "../checkpoints/best_model.pth",
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    # If model not found, try downloading from URL (if configured)
    if not model_path:
        # Check for model URL in secrets or environment
        model_url = os.environ.get('MODEL_URL') or st.secrets.get('MODEL_URL', None)
        if model_url:
            default_path = "checkpoints/best_model.pth"
            if download_model_from_url(model_url, default_path):
                model_path = default_path
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, device
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
    
    # Check if we should use demo mode
    use_demo_mode = model is None
    if use_demo_mode:
        st.warning("🎭 **Demo Mode**: Model not found. Using simulated predictions for demonstration.")
        st.info("💡 To use real predictions, train the model and add `checkpoints/best_model.pth`")
    
    # Demo mode prediction function
    def create_demo_prediction(image_array, image_size=(256, 256)):
        """Create demo prediction when model is not available."""
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        threshold = np.percentile(normalized, 75)
        mask = np.zeros(image_size, dtype=np.int32)
        mask[(normalized >= threshold * 0.7) & (normalized < threshold * 1.2)] = 1
        mask[normalized >= threshold * 1.2] = 2
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
        mask = np.round(mask).astype(np.int32)
        prob_bg = np.mean(mask == 0)
        prob_benign = np.mean(mask == 1)
        prob_malignant = np.mean(mask == 2)
        probabilities = np.array([
            np.full(image_size, prob_bg),
            np.full(image_size, prob_benign),
            np.full(image_size, prob_malignant)
        ])
        return mask, probabilities
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Check if test images are available
        test_dir = "data/test"
        example_images = []
        if os.path.exists(test_dir):
            example_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            example_images.sort()
        
        # Example image selector
        if example_images:
            st.info(f"💡 **{len(example_images)} example images available!** Try one below or upload your own.")
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                if st.button("📷 Try Example Image 1", use_container_width=True):
                    example_path = os.path.join(test_dir, example_images[0])
                    st.session_state['use_example'] = example_path
            with col_ex2:
                if len(example_images) > 1 and st.button("📷 Try Example Image 2", use_container_width=True):
                    example_path = os.path.join(test_dir, example_images[1])
                    st.session_state['use_example'] = example_path
        
        uploaded_file = st.file_uploader(
            "Or upload your own breast ultrasound image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color breast ultrasound image"
        )
        
        # Handle example image
        if 'use_example' in st.session_state and st.session_state['use_example']:
            example_path = st.session_state['use_example']
            if os.path.exists(example_path):
                uploaded_file = open(example_path, 'rb')
                st.success(f"✅ Using example: {os.path.basename(example_path)}")
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Determine caption
            if 'use_example' in st.session_state and st.session_state['use_example']:
                caption = f"Example Image: {os.path.basename(st.session_state['use_example'])}"
            else:
                caption = "Uploaded Image"
            
            st.image(image, caption=caption, use_container_width=True)
            
            # Process and predict
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    if use_demo_mode:
                        # Use demo mode prediction
                        prediction, probabilities = create_demo_prediction(image_array)
                        processed_image = cv2.resize(
                            cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array,
                            (256, 256)
                        ).astype(np.float32) / 255.0
                    else:
                        # Use real model prediction
                        image_tensor, processed_image = preprocess_image(image_array)
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
                    # Clear example flag after use
                    if 'use_example' in st.session_state:
                        del st.session_state['use_example']
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
            st.info("👆 **Try it now!**")
            st.markdown("""
            **Option 1:** Click "Try Example Image" buttons above to test with sample images  
            **Option 2:** Upload your own breast ultrasound image  
            **Option 3:** Upload any image to see how the segmentation works
            """)
            
            # Show preview of what to expect
            if example_images:
                st.markdown("---")
                st.markdown("### 📸 Sample Images Available")
                # Show thumbnails of first few examples
                cols = st.columns(min(3, len(example_images)))
                for idx, img_name in enumerate(example_images[:3]):
                    with cols[idx % 3]:
                        img_path = os.path.join(test_dir, img_name)
                        if os.path.exists(img_path):
                            try:
                                preview_img = Image.open(img_path)
                                # Resize for thumbnail
                                preview_img.thumbnail((150, 150))
                                st.image(preview_img, caption=img_name, use_container_width=True)
                            except:
                                pass
    
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

