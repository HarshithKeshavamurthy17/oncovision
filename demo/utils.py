"""
Utility functions for demo deployment.
"""
import os
import urllib.request
from pathlib import Path

def download_model(model_url=None, model_path="checkpoints/best_model.pth"):
    """
    Download model from URL if it doesn't exist locally.
    
    Args:
        model_url: URL to download model from (optional)
        model_path: Local path to save model
    """
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return True
    
    if model_url is None:
        print("Model not found and no URL provided.")
        print("Please train the model first or provide a download URL.")
        return False
    
    print(f"Downloading model from {model_url}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def check_model_exists(model_path="checkpoints/best_model.pth"):
    """Check if model file exists."""
    return os.path.exists(model_path)




