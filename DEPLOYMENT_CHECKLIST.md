# Railway Deployment Checklist

Complete checklist to ensure successful deployment of OncoVision to Railway.

## ✅ Pre-Deployment Checklist

### Code Preparation
- [ ] All code is committed and pushed to GitHub
- [ ] `demo/app.py` exists and is the main entry point
- [ ] `demo/requirements.txt` exists with all dependencies
- [ ] All dependencies use `>=` (not `==`) for version flexibility
- [ ] No hardcoded paths that won't work on Railway
- [ ] App handles missing model file gracefully (demo mode)

### Requirements.txt Verification
- [ ] `streamlit>=1.28.0`
- [ ] `torch>=2.0.0`
- [ ] `torchvision>=0.15.0`
- [ ] `segmentation-models-pytorch>=0.3.3`
- [ ] `opencv-python-headless>=4.8.0`
- [ ] `Pillow>=10.0.0`
- [ ] `numpy>=1.24.0`
- [ ] `pandas>=2.0.0`
- [ ] `albumentations>=1.3.0`
- [ ] `matplotlib>=3.7.0`
- [ ] `scikit-learn>=1.3.0`
- [ ] `tqdm>=4.65.0`
- [ ] All packages use `>=` (flexible versions)

### Local Testing
- [ ] App runs locally: `streamlit run demo/app.py`
- [ ] No import errors
- [ ] App loads without crashes
- [ ] Can upload images and get results (or demo mode works)

## 🚀 Railway Setup Checklist

### Account & Project
- [ ] Created Railway account at [railway.app](https://railway.app)
- [ ] Signed in with GitHub account
- [ ] Created new project
- [ ] Connected GitHub repository
- [ ] Selected correct repository branch (usually `main`)

### Configuration
- [ ] Set Root Directory (if app is in `demo/`, set to `demo`)
- [ ] Set Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - OR if root is `/`: `streamlit run demo/app.py --server.port $PORT --server.address 0.0.0.0`
- [ ] Start command includes `$PORT` (required)
- [ ] Start command includes `0.0.0.0` (required for external access)
- [ ] Python version is compatible (Railway auto-detects)

### Environment Variables (Optional)
- [ ] Added `PYTHONUNBUFFERED=1` (recommended for better logs)
- [ ] Added `MODEL_URL` if using remote model download
- [ ] Any other required environment variables set

### Domain & Networking
- [ ] Generated Railway domain in Settings → Networking
- [ ] Copied domain URL (e.g., `oncovision-production.up.railway.app`)
- [ ] Domain is accessible

## 🔨 Build & Deploy Checklist

### Build Process
- [ ] Railway detected Python project automatically
- [ ] Build started automatically (or manually triggered)
- [ ] Build logs show "Installing dependencies"
- [ ] Build logs show "Build successful" (no errors)
- [ ] No "ModuleNotFoundError" in build logs
- [ ] No "Failed to build wheel" errors

### Deployment Process
- [ ] Deployment started after successful build
- [ ] Deployment logs show "Starting application"
- [ ] Deployment logs show "You can now view your Streamlit app"
- [ ] Deployment status shows green checkmark (success)
- [ ] No crashes in deployment logs

## ✅ Post-Deployment Verification

### App Accessibility
- [ ] Can access app at Railway domain URL
- [ ] App loads (may take 10-30 seconds if sleeping)
- [ ] No "Connection refused" errors
- [ ] No "502 Bad Gateway" errors
- [ ] App UI displays correctly

### Functionality Testing
- [ ] Can see Streamlit interface
- [ ] Sidebar displays correctly
- [ ] Can upload an image (or use example images)
- [ ] "Analyze Image" button works
- [ ] Segmentation results display (or demo mode works)
- [ ] No runtime errors in Railway logs
- [ ] App handles errors gracefully

### Performance Check
- [ ] App loads within reasonable time (< 30 seconds)
- [ ] Image processing works (even if slow on CPU)
- [ ] No memory errors in logs
- [ ] App doesn't crash on multiple requests

## 📝 Documentation Updates

### README.md
- [ ] Updated with Railway deployment section
- [ ] Added live demo URL (Railway domain)
- [ ] Added note about free tier wake-up time (10-30 seconds)
- [ ] Removed/updated old deployment URLs if any
- [ ] Added link to Railway deployment guide

### Other Files
- [ ] `RAILWAY_DEPLOYMENT.md` created with deployment steps
- [ ] `TROUBLESHOOTING.md` created with common issues
- [ ] `DEPLOYMENT_CHECKLIST.md` created (this file)

## 🔄 Continuous Deployment

### Auto-Deploy Setup
- [ ] Railway connected to GitHub repository
- [ ] Auto-deploy enabled (default)
- [ ] Pushing to main branch triggers deployment
- [ ] Verified by making a small change and pushing

### Monitoring
- [ ] Know how to check Railway logs
- [ ] Know how to view deployment history
- [ ] Understand Railway dashboard layout
- [ ] Know where to find metrics

## 🎯 Final Verification

### Before Sharing
- [ ] App is fully functional
- [ ] All features work as expected
- [ ] No critical errors in logs
- [ ] README has correct live demo URL
- [ ] Documentation is complete

### Portfolio Integration
- [ ] Added live demo link to portfolio
- [ ] Screenshots/video of working app (optional)
- [ ] GitHub repository link included
- [ ] Tech stack mentioned

## 🐛 If Something Goes Wrong

### Quick Debug Steps
1. [ ] Check Railway build logs for errors
2. [ ] Check Railway runtime logs for errors
3. [ ] Verify start command is correct
4. [ ] Verify all dependencies in requirements.txt
5. [ ] Test app locally to compare
6. [ ] Check troubleshooting guide
7. [ ] Check Railway status page

### Common Issues Reference
- See `TROUBLESHOOTING.md` for detailed solutions
- Check Railway documentation
- Ask in Railway Discord if needed

## 📊 Success Criteria

Your deployment is successful when:
- ✅ App is accessible at Railway URL
- ✅ App loads and displays correctly
- ✅ Can upload/process images
- ✅ No errors in Railway logs
- ✅ Auto-deploy works (push to GitHub = new deployment)
- ✅ Documentation is updated

## 🎉 You're Done!

Once all items are checked:
1. ✅ Share your live demo URL
2. ✅ Add to portfolio
3. ✅ Celebrate! 🎊

---

**Remember:**
- Free tier apps sleep after 7 days - this is normal
- Wake-up takes 10-30 seconds - set expectations in README
- Auto-deploy means just push to GitHub for updates
- Check logs if anything goes wrong

**Need Help?**
- See `TROUBLESHOOTING.md`
- See `RAILWAY_DEPLOYMENT.md`
- Check Railway docs: [docs.railway.app](https://docs.railway.app)

