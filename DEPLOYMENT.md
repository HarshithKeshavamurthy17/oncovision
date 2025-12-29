# Deployment Guide for OncoVision Demo

This guide explains how to deploy the OncoVision demo for your portfolio.

## Demo Options

### Option 1: Streamlit Cloud (Recommended - Easiest)

**Best for:** Quick deployment, free hosting, easy updates

1. **Prepare your repository:**
   - Ensure all code is pushed to GitHub
   - Create a `requirements.txt` in the root with all dependencies
   - Ensure `demo/app.py` exists

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `HarshithKeshavamurthy17/oncovision`
   - Set Main file path: `demo/app.py`
   - Click "Deploy"

3. **Add model weights:**
   - Upload `best_model.pth` to a cloud storage (Google Drive, Dropbox)
   - Or use GitHub Releases/LFS for the model file
   - Update the code to download the model on first run

4. **Your demo will be live at:**
   ```
   https://your-username-oncovision.streamlit.app
   ```

### Option 2: Gradio Spaces (Alternative)

**Best for:** Hugging Face integration, easy sharing

1. **Create Hugging Face Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create new Space
   - Choose Gradio SDK
   - Clone the space repository

2. **Add files:**
   - Copy `demo/gradio_app.py` to the space
   - Copy model weights
   - Add requirements.txt

3. **Deploy:**
   - Push to the space repository
   - Auto-deploys!

4. **Your demo will be live at:**
   ```
   https://huggingface.co/spaces/your-username/oncovision
   ```

### Option 3: Heroku (More Control)

1. **Install Heroku CLI**
2. **Create Procfile:**
   ```
   web: streamlit run demo/app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. **Deploy:**
   ```bash
   heroku create oncovision-demo
   git push heroku main
   ```

### Option 4: Local Hosting (For Testing)

```bash
# Install dependencies
pip install streamlit

# Run the demo
streamlit run demo/app.py

# Or with Gradio
pip install gradio
python demo/gradio_app.py
```

## Adding Model to Repository

Since model files are large, you have several options:

### Option A: GitHub Releases
1. Create a release on GitHub
2. Upload `best_model.pth` as an asset
3. Update code to download from release

### Option B: Google Drive / Dropbox
1. Upload model to cloud storage
2. Share public link
3. Update code to download on first run

### Option C: Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add checkpoints/best_model.pth
git commit -m "Add model with LFS"
git push
```

## Updating Demo Code for Deployment

For production deployment, you might want to add model downloading:

```python
import urllib.request
import os

MODEL_URL = "https://github.com/HarshithKeshavamurthy17/oncovision/releases/download/v1.0/best_model.pth"

def download_model():
    if not os.path.exists("checkpoints/best_model.pth"):
        os.makedirs("checkpoints", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, "checkpoints/best_model.pth")
```

## Embedding in Portfolio

### Iframe Embed (Streamlit)
```html
<iframe 
    src="https://your-username-oncovision.streamlit.app" 
    width="100%" 
    height="800px" 
    frameborder="0">
</iframe>
```

### Direct Link Button
```html
<a href="https://your-username-oncovision.streamlit.app" 
   target="_blank" 
   class="demo-button">
   🚀 Try Live Demo
</a>
```

### Screenshots
- Take screenshots of the demo
- Add to portfolio with link
- Shows functionality without requiring hosting

## Environment Variables

For sensitive configurations, use environment variables:

```bash
export MODEL_PATH="checkpoints/best_model.pth"
export MAX_IMAGE_SIZE=512
```

## Troubleshooting

### Model Not Found Error
- Ensure model file is accessible
- Check file paths
- Use absolute paths in production

### Memory Issues
- Reduce batch size
- Use CPU instead of GPU if needed
- Optimize image preprocessing

### Slow Loading
- Cache model loading with `@st.cache_resource`
- Pre-process images client-side
- Use smaller model for demo

## Portfolio Integration Tips

1. **Add Screenshots:** Show demo UI before linking
2. **Video Demo:** Record a short demo video
3. **Feature List:** Highlight key features
4. **GitHub Link:** Always include source code link
5. **Tech Stack:** Mention technologies used

## Quick Start Commands

```bash
# Test locally
streamlit run demo/app.py

# Deploy to Streamlit Cloud
# Just push to GitHub and connect via share.streamlit.io

# Test Gradio
python demo/gradio_app.py
```




