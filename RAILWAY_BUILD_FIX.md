# Railway Build Timeout Fix

## 🐛 Problem
Build is timing out because Railway is installing PyTorch with CUDA libraries, which are:
- **Huge** (several GB)
- **Unnecessary** (Railway free tier is CPU-only)
- **Slow to install** (causes build timeout)

## ✅ Solution

### Option 1: Set Root Directory to `demo` (RECOMMENDED)

**In Railway Dashboard:**
1. Go to **Settings** → **Root Directory**
2. Set to: `demo`
3. This makes Railway use `demo/requirements.txt` (optimized for deployment)
4. **Redeploy** - the build should be much faster

### Option 2: Use Optimized Root Requirements

If you want to keep root directory as `/`, the root `requirements.txt` has been optimized with version constraints to prevent installing CUDA libraries.

## 🔧 What Was Fixed

1. **Removed tensorboard** from root requirements (not needed for Streamlit app)
2. **Added version constraints** to prevent numpy 2.x (which can cause compatibility issues)
3. **Added version constraints** to PyTorch to prevent latest versions that might pull CUDA

## 📋 Railway Settings Checklist

Make sure these are set correctly:

- [ ] **Root Directory:** `demo` (RECOMMENDED) or `/`
- [ ] **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - OR if root is `/`: `streamlit run demo/app.py --server.port $PORT --server.address 0.0.0.0`
- [ ] **Requirements file:** Railway will use `requirements.txt` in the root directory you set

## 🚀 After Fixing

1. **Commit and push the changes:**
   ```bash
   git add requirements.txt demo/requirements.txt
   git commit -m "Optimize requirements.txt for Railway CPU-only build"
   git push origin main
   ```

2. **In Railway:**
   - Set Root Directory to `demo` (if not already)
   - Railway will auto-redeploy
   - Build should complete in 2-3 minutes instead of timing out

## 💡 Why This Works

- **CPU-only PyTorch** is much smaller (~500MB vs ~3GB with CUDA)
- **Version constraints** prevent pulling latest versions that might have CUDA
- **Smaller requirements** = faster build = no timeout

## ⚠️ If Still Timing Out (AGGRESSIVE FIX)

If build is still timing out after setting root directory to `demo`, try this:

### Option 3: Use Custom Build Command

1. In Railway **Settings** → **Build Command**, set:
   ```bash
   pip install --no-cache-dir -r demo/requirements.txt
   ```

2. This forces Railway to use the optimized `demo/requirements.txt`

### Option 4: Remove Heavy Dependencies Temporarily

Edit `demo/requirements.txt` and comment out:
- `gradio>=4.0.0` (if not using Gradio interface)
- This saves significant build time

### Option 5: Use Requirements File with Explicit CPU Index

The `--extra-index-url https://download.pytorch.org/whl/cpu` must be at the **very top** of requirements.txt (before any packages).

### Option 6: Set Build Command to Use Railway-Optimized File

1. Copy `demo/requirements-railway.txt` to root as `requirements.txt`
2. Or set custom build command:
   ```bash
   pip install --no-cache-dir -r demo/requirements-railway.txt
   ```

## 🔍 Debugging

Check build logs for:
- Are CUDA libraries still being installed? (nvidia-* packages)
- Is Railway using the correct requirements file?
- Is the `--extra-index-url` at the top of the file?

---

**CRITICAL:** Make sure Root Directory is set to `demo` in Railway Settings!

