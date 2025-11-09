"""
Streamlit Demo App for OncoVision - DEMO MODE
This version works without a trained model for portfolio showcase.
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

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
    .demo-badge {
        background-color: #ff9800;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.9em;
        display: inline-block;
        margin-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def create_demo_prediction(image_array, image_size=(256, 256)):
    """
    Create a demo prediction mask for showcase purposes.
    This simulates what the model would predict.
    """
    # Resize image
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    
    # Create a simple demo mask based on image intensity
    # This simulates finding regions of interest
    normalized = resized.astype(np.float32) / 255.0
    
    # Create demo segmentation
    # Simulate finding bright regions (potential tumors)
    threshold = np.percentile(normalized, 75)
    mask = np.zeros(image_size, dtype=np.int32)
    
    # Background (0)
    mask[normalized < threshold * 0.7] = 0
    
    # Simulate benign regions (1) - medium brightness
    benign_mask = (normalized >= threshold * 0.7) & (normalized < threshold * 1.2)
    mask[benign_mask] = 1
    
    # Simulate malignant regions (2) - high brightness
    malignant_mask = normalized >= threshold * 1.2
    mask[malignant_mask] = 2
    
    # Add some smoothing
    mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
    mask = np.round(mask).astype(np.int32)
    
    # Create probabilities (demo values)
    prob_bg = np.mean(mask == 0)
    prob_benign = np.mean(mask == 1)
    prob_malignant = np.mean(mask == 2)
    
    probabilities = np.array([
        np.full(image_size, prob_bg),
        np.full(image_size, prob_benign),
        np.full(image_size, prob_malignant)
    ])
    
    return mask, probabilities, resized

def create_visualization(original_image, prediction, probabilities):
    """Create visualization of the prediction."""
    # Create colored mask
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    
    # Color mapping: Background=black, Benign=green, Malignant=red
    colored_mask[prediction == 1] = [0, 255, 0]  # Benign - Green
    colored_mask[prediction == 2] = [255, 0, 0]  # Malignant - Red
    
    # Overlay on original image
    if len(original_image.shape) == 2:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_image
    
    overlay = cv2.addWeighted(
        (original_rgb * 255).astype(np.uint8) if original_image.dtype == np.float32 else original_rgb,
        0.6,
        colored_mask,
        0.4,
        0
    )
    
    return colored_mask, overlay

def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">🔬 OncoVision <span class="demo-badge">DEMO MODE</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning-based Breast Ultrasound Image Segmentation</p>', unsafe_allow_html=True)
    
    st.info("🎭 **Demo Mode**: This is a showcase version with simulated predictions. For real predictions, train the model and use the full version.")
    
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
        This is a **DEMO MODE** with simulated predictions.
        For production use, train the model first.
        Not intended for clinical diagnosis.
        """)
        
        st.header("🔗 Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/HarshithKeshavamurthy17/oncovision)
        - [Documentation](https://github.com/HarshithKeshavamurthy17/oncovision#readme)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a breast ultrasound image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a grayscale or color breast ultrasound image"
        )
        
        # Try to load example from test data
        example_loaded = False
        if not uploaded_file:
            test_dir = "data/test"
            if os.path.exists(test_dir):
                test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if test_images:
                    st.info(f"💡 **Tip**: {len(test_images)} example images available in test folder")
                    if st.button("📷 Load Example Image"):
                        example_path = os.path.join(test_dir, test_images[0])
                        uploaded_file = open(example_path, 'rb')
                        example_loaded = True
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process and predict
            if st.button("🔍 Analyze Image", type="primary") or example_loaded:
                with st.spinner("Processing image..."):
                    # Create demo prediction
                    prediction, probabilities, processed_image = create_demo_prediction(image_array)
                    
                    # Visualize
                    colored_mask, overlay = create_visualization(
                        processed_image / 255.0, prediction, probabilities
                    )
                    
                    # Store in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['probabilities'] = probabilities
                    st.session_state['overlay'] = overlay
                    st.session_state['mask'] = colored_mask
                    st.session_state['original'] = processed_image
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            # Display overlay
            st.image(st.session_state['overlay'], caption="Segmentation Overlay (Demo)", use_container_width=True)
            
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
                         delta="Detected" if benign_pixels > 0 else None)
            with col_c:
                st.metric("Malignant", f"{malignant_pixels/total_pixels*100:.1f}%",
                         delta="⚠️ Detected" if malignant_pixels > 0 else None,
                         delta_color="inverse" if malignant_pixels > 0 else "normal")
            
            # Probability distribution
            st.subheader("🎯 Class Probabilities")
            prob_dict = {
                "Background": float(probabilities[0].mean()),
                "Benign": float(probabilities[1].mean()),
                "Malignant": float(probabilities[2].mean())
            }
            st.bar_chart(prob_dict)
            
            st.info("💡 **Note**: These are simulated predictions for demonstration purposes.")
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
            st.info("🎭 **Demo Mode**: Predictions are simulated based on image intensity patterns.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using PyTorch, Streamlit, and Segmentation Models</p>
        <p>© 2024 OncoVision Project | Demo Mode</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

