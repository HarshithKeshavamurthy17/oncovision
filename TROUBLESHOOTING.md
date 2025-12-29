# Troubleshooting Guide for Railway Deployment

Common errors and their solutions when deploying OncoVision to Railway.

## 🔴 Build Failures

### Error: "No matching distribution found for package==X.X.X"

**Problem:** Exact version pinning (`==`) causes build failures when that version isn't available.

**Solution:**
- Change all `==` to `>=` in `requirements.txt`
- Example: `torch==2.0.0` → `torch>=2.0.0`
- Railway's build environment may have different package versions available

**Fix:**
```bash
# Check requirements.txt
# Replace all == with >=
sed -i 's/==/>=/g' demo/requirements.txt
```

### Error: "Failed to build wheel for package"

**Problem:** Package requires compilation and build tools are missing.

**Solution:**
- Ensure you're using pre-built wheels when possible
- For packages like `opencv-python-headless`, use the `-headless` version (already in requirements)
- Railway should handle most builds automatically

**Check:**
- `opencv-python-headless` (not `opencv-python`) ✓
- All packages use `>=` for flexibility ✓

### Error: "ERROR: Could not find a version that satisfies the requirement"

**Problem:** Package name is misspelled or doesn't exist.

**Solution:**
- Verify package names in `requirements.txt`
- Check PyPI for correct package names
- Common mistakes:
  - `opencv` → `opencv-python-headless` ✓
  - `PIL` → `Pillow` ✓
  - `cv2` → `opencv-python-headless` ✓

## 🟡 Runtime Errors

### Error: "ModuleNotFoundError: No module named 'X'"

**Problem:** Missing dependency in `requirements.txt`.

**Solution:**
1. Check all imports in `demo/app.py`:
   ```python
   import streamlit
   import torch
   import numpy
   import cv2
   from PIL import Image
   import segmentation_models_pytorch
   ```

2. Verify all are in `requirements.txt`:
   - `streamlit>=1.28.0` ✓
   - `torch>=2.0.0` ✓
   - `numpy>=1.24.0` ✓
   - `opencv-python-headless>=4.8.0` ✓
   - `Pillow>=10.0.0` ✓
   - `segmentation-models-pytorch>=0.3.3` ✓

3. Add any missing dependencies

### Error: "Address already in use" or Port Issues

**Problem:** Start command doesn't use `$PORT` environment variable.

**Solution:**
- Ensure start command uses `$PORT`:
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
  ```
- Railway sets `$PORT` automatically - you must use it
- `0.0.0.0` allows external connections

**Check Railway Settings:**
- Settings → Deploy → Start Command
- Must include `$PORT` and `0.0.0.0`

### Error: "Connection refused" or "Cannot connect"

**Problem:** App not binding to correct address/port.

**Solution:**
- Verify start command has both:
  - `--server.port $PORT` (uses Railway's port)
  - `--server.address 0.0.0.0` (allows external access)

**Correct command:**
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 🟠 Application Crashes

### Error: "Model not found" or FileNotFoundError

**Problem:** Model file path is incorrect or file doesn't exist.

**Solution:**
1. **Check model paths in code:**
   ```python
   # In demo/app.py, model_paths includes:
   "checkpoints/best_model.pth"
   "../checkpoints/best_model.pth"
   os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pth")
   ```

2. **Options:**
   - Upload model to GitHub (if small enough)
   - Use MODEL_URL environment variable to download
   - Use demo mode (app already handles this gracefully)

3. **App will use demo mode if model not found** - this is fine for testing!

### Error: "Out of memory" or Memory Issues

**Problem:** App uses too much memory on Railway free tier.

**Solution:**
- Railway free tier has memory limits
- Optimize model loading:
  - Use `@st.cache_resource` (already in code) ✓
  - Load model only once
  - Use CPU instead of GPU if needed

**Check:**
- Model is cached with `@st.cache_resource` ✓
- App handles missing model gracefully (demo mode) ✓

### Error: "ImportError: cannot import name 'X'"

**Problem:** Import path issue or missing `__init__.py`.

**Solution:**
1. **Check import paths:**
   ```python
   # In demo/app.py:
   sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
   from src.model import get_enhanced_unet
   ```

2. **Verify file structure:**
   ```
   oncovision/
   ├── src/
   │   ├── __init__.py  # Must exist
   │   ├── model.py
   │   └── config.py
   └── demo/
       └── app.py
   ```

3. **Ensure `src/__init__.py` exists** (should be empty or have package init)

## 🟢 Common Warnings (Usually Safe to Ignore)

### Warning: "Using CPU instead of GPU"

**Status:** ✅ Normal and expected

**Explanation:**
- Railway free tier doesn't provide GPU
- App automatically uses CPU (code handles this)
- Performance may be slower but works fine

### Warning: "Model file not found, using demo mode"

**Status:** ✅ Normal if model not uploaded

**Explanation:**
- App gracefully falls back to demo mode
- Demo mode uses image processing instead of ML model
- Still functional for demonstration purposes

## 🔍 How to Check Railway Logs

### View Build Logs:
1. Go to Railway dashboard
2. Click your service
3. Go to **Deployments** tab
4. Click latest deployment
5. View **Build Logs** section

### View Runtime Logs:
1. Same as above
2. View **Logs** tab (runtime logs)
3. Real-time logs show app output

### Common Log Messages:

**Good signs:**
- ✅ "Build successful"
- ✅ "You can now view your Streamlit app"
- ✅ "Network URL: http://0.0.0.0:PORT"

**Warning signs:**
- ⚠️ "ModuleNotFoundError" → Missing dependency
- ⚠️ "Address already in use" → Port configuration issue
- ⚠️ "FileNotFoundError" → Missing file/path issue

## 🛠️ Quick Fixes

### Fix 1: Update requirements.txt
```bash
# Ensure all use >=
# Check for missing packages
# Verify package names
```

### Fix 2: Fix Start Command
```bash
# In Railway Settings → Deploy → Start Command:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Fix 3: Check Root Directory
- If app is in `demo/`, set Root Directory to `demo`
- Or adjust start command to `demo/app.py`

### Fix 4: Add Missing Dependencies
```bash
# Check imports in app.py
# Add to requirements.txt if missing
# Push to GitHub (auto-deploys)
```

## 📋 Debugging Checklist

When app doesn't work:

- [ ] Check Railway build logs for errors
- [ ] Verify start command uses `$PORT` and `0.0.0.0`
- [ ] Check all dependencies in `requirements.txt`
- [ ] Verify root directory is correct
- [ ] Check runtime logs for import errors
- [ ] Verify file paths are correct
- [ ] Test locally first: `streamlit run demo/app.py`
- [ ] Check Python version compatibility

## 🆘 Still Having Issues?

1. **Check Railway Status:** [status.railway.app](https://status.railway.app)
2. **Railway Discord:** [discord.gg/railway](https://discord.gg/railway)
3. **Railway Docs:** [docs.railway.app](https://docs.railway.app)
4. **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)

## 💡 Pro Tips

1. **Test locally first:**
   ```bash
   streamlit run demo/app.py
   ```
   If it works locally, it should work on Railway

2. **Use demo mode:**
   - App works without model file
   - Good for testing deployment
   - Add model later if needed

3. **Check logs immediately:**
   - Railway logs show errors quickly
   - Most issues are visible in first few log lines

4. **Incremental debugging:**
   - Fix one error at a time
   - Push changes and check logs
   - Repeat until working

---

**Remember:** Most deployment issues are:
- Missing dependencies → Add to requirements.txt
- Wrong start command → Use `$PORT` and `0.0.0.0`
- Path issues → Check root directory and file paths

