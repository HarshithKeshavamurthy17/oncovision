# Railway Deployment Guide for OncoVision

Complete step-by-step guide to deploy your OncoVision Streamlit application to Railway.app.

## 📋 Prerequisites

- GitHub account
- Railway account (sign up at [railway.app](https://railway.app))
- Your OncoVision project pushed to GitHub
- $5 monthly credit (Railway free tier)

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure your code is on GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Verify your project structure:**
   - Main app file: `demo/app.py`
   - Requirements file: `demo/requirements.txt`
   - All dependencies use `>=` (not `==`) for version flexibility

### Step 2: Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Sign in with your **GitHub account** (recommended for easy integration)
4. Railway will give you $5 in free credits monthly

### Step 3: Create a New Project

1. In Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository: `oncovision` (or your repo name)
4. Railway will automatically detect it's a Python project

### Step 4: Configure Your Service

1. **Set the Root Directory:**
   - Go to **Settings** → **Root Directory**
   - Set to: `demo` (since your app is in the demo folder)
   - OR keep it as root and adjust the start command (see below)

2. **Set the Start Command:**
   - Go to **Settings** → **Deploy** → **Start Command**
   - Enter:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - If you kept root directory as `/`, use:
     ```bash
     streamlit run demo/app.py --server.port $PORT --server.address 0.0.0.0
     ```

3. **Set Python Version (Optional):**
   - Railway auto-detects Python, but you can specify in `runtime.txt`:
     ```
     python-3.11
     ```
   - Or Railway will use the latest compatible version

### Step 5: Configure Requirements File Location

1. **If using `demo/` as root directory:**
   - Railway will automatically find `requirements.txt` in the root directory
   - Make sure `demo/requirements.txt` exists and has all dependencies

2. **If using project root:**
   - Railway will use `requirements.txt` in the root
   - You may need to copy `demo/requirements.txt` to root, or
   - Specify custom build command (not recommended)

### Step 6: Set Environment Variables (Optional)

If your app needs environment variables:

1. Go to **Variables** tab in Railway
2. Add any required variables:
   - `MODEL_URL`: URL to download model weights (if using remote model)
   - `PYTHONUNBUFFERED=1`: For better logging (recommended)

### Step 7: Generate Domain

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Railway will create a domain like: `oncovision-production.up.railway.app`
4. Copy this URL - this is your live app URL!

### Step 8: Deploy

1. Railway will automatically start building when you:
   - Push to your connected GitHub branch
   - Or click **"Deploy"** in the Railway dashboard

2. **Monitor the build:**
   - Go to **Deployments** tab
   - Watch the build logs
   - Wait for "Build successful" message

3. **Check deployment status:**
   - Green checkmark = deployed successfully
   - Click on the deployment to see logs

### Step 9: Verify Deployment

1. **Visit your domain:**
   - Open the Railway-generated URL
   - Wait 10-30 seconds if app was sleeping (free tier behavior)

2. **Test the app:**
   - Upload an image or use example images
   - Verify segmentation works
   - Check for any errors in Railway logs

## 🔧 Configuration Summary

### Recommended Settings:

- **Root Directory:** `demo` (or leave as root)
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Build Command:** (Auto-detected, no need to set)
- **Python Version:** Auto-detected (or specify in `runtime.txt`)

### File Structure Railway Expects:

```
oncovision/
├── demo/
│   ├── app.py              # Main Streamlit app
│   ├── requirements.txt    # Dependencies
│   └── ...
├── src/                    # Source code
├── data/                   # Data files (if needed)
└── README.md
```

## 🔄 Auto-Deployment

Railway automatically deploys when you:
- Push to the connected GitHub branch (usually `main` or `master`)
- Merge pull requests
- Make commits to the branch

**No manual redeploy needed!** Just push to GitHub and Railway handles the rest.

## 💰 Free Tier Information

### What You Get:
- **$5 monthly credit** (enough for small apps)
- **Automatic deployments** from GitHub
- **Custom domain** support
- **Sleep mode** after 7 days of inactivity

### Sleep Mode Behavior:
- App sleeps after **7 days** of no activity
- Takes **10-30 seconds** to wake up on first visit
- This is **normal and expected** for free tier
- No action needed - just wait for wake-up

### Important Notes:
- ⚠️ **Don't create keep-alive scripts** - violates hosting terms
- ✅ **Accept sleep mode** as normal free tier behavior
- ✅ **Set expectations** in README about wake-up time
- ✅ **Free tier is perfect** for portfolio projects

## 📝 Post-Deployment Checklist

- [ ] App loads successfully at Railway URL
- [ ] Can upload images and get predictions
- [ ] No errors in Railway logs
- [ ] Domain is accessible
- [ ] Updated README with live demo link
- [ ] Tested on mobile (if applicable)

## 🔍 Monitoring Your App

### View Logs:
1. Go to Railway dashboard
2. Click on your service
3. Go to **Deployments** tab
4. Click on latest deployment
5. View **Logs** tab for real-time logs

### Check Metrics:
- **Deployments** tab: See deployment history
- **Metrics** tab: CPU, memory usage
- **Settings** tab: Configuration

## 🐛 Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues and solutions.

Common issues:
- Build failures → Check requirements.txt
- App crashes → Check start command
- Port errors → Ensure `$PORT` is used
- Import errors → Check all dependencies listed

## 🔗 Next Steps

1. **Update README.md** with your Railway URL
2. **Add to portfolio** with live demo link
3. **Share with others** - your app is live!
4. **Monitor usage** in Railway dashboard

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud)
- [Railway Discord](https://discord.gg/railway) for support

---

**Need Help?** Check the troubleshooting guide or Railway's documentation.

