# Model Setup Guide

To make your OncoVision demo fully functional, you need to add a trained model file.

## Option 1: Train the Model (Recommended)

Train the model using your dataset:

```bash
# Make sure you have the data organized
data/
├── train/
│   ├── benign/
│   ├── malignant/
│   └── normal/

# Train the model
python -m src.train
```

This will create `checkpoints/best_model.pth` which you can then add to your repository.

## Option 2: Add Model via Git LFS

For large model files (>100MB), use Git LFS:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Track .pth files
git lfs track "*.pth"
git add .gitattributes

# Add your model
git add checkpoints/best_model.pth
git commit -m "Add trained model"
git push
```

## Option 3: GitHub Releases (For Large Models)

1. **Train your model locally** (or use existing trained model)

2. **Create a GitHub Release:**
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Tag: `v1.0.0`
   - Upload `best_model.pth` as a release asset

3. **Configure automatic download:**
   - In Streamlit Cloud, go to Settings → Secrets
   - Add:
   ```
   MODEL_URL = "https://github.com/HarshithKeshavamurthy17/oncovision/releases/download/v1.0.0/best_model.pth"
   ```

4. **Or update the code** to use the release URL directly

## Option 4: Google Drive / Dropbox

1. Upload `best_model.pth` to Google Drive
2. Get a shareable link (make it public)
3. Convert to direct download link:
   - For Google Drive: Replace `/file/d/FILE_ID/view` with `/uc?export=download&id=FILE_ID`
4. Add to Streamlit Secrets:
   ```
   MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
   ```

## Option 5: Quick Test Model (For Demo Only)

If you just want to test the interface without full training, you can create a minimal placeholder. However, this won't provide real predictions.

**Note:** For a production portfolio demo, you should use a properly trained model.

## Streamlit Cloud Secrets Configuration

1. Go to your Streamlit Cloud app dashboard
2. Click "Settings" → "Secrets"
3. Add:
   ```toml
   MODEL_URL = "your_model_download_url_here"
   ```
4. Save and redeploy

## Verification

After adding the model:

1. The demo should automatically load the model
2. You can upload test images
3. Get real segmentation predictions

## Troubleshooting

### Model file too large for GitHub
- Use Git LFS (Option 2)
- Use GitHub Releases (Option 3)
- Use cloud storage (Option 4)

### Model not loading
- Check file path is correct
- Verify model file is not corrupted
- Check Streamlit logs for errors

### Download fails
- Verify URL is accessible
- Check URL format (should be direct download link)
- Try downloading manually to test

## Model File Size

Typical model sizes:
- **Small**: 50-100 MB (compressed)
- **Medium**: 100-300 MB
- **Large**: 300+ MB

Streamlit Cloud can handle models up to 1GB, but larger files may require Git LFS or external storage.




