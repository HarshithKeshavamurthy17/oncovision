# Railway Quick Start Guide

**Quick reference for deploying OncoVision to Railway in 5 minutes.**

## 🚀 Quick Deploy Steps

1. **Sign up:** [railway.app](https://railway.app) → Sign in with GitHub

2. **Create project:** New Project → Deploy from GitHub repo → Select `oncovision`

3. **Configure:**
   - **Root Directory:** `demo` (or leave as root)
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Generate domain:** Settings → Networking → Generate Domain

5. **Done!** Your app is live at `https://oncovision-production.up.railway.app`

## ⚙️ Essential Settings

```
Root Directory: demo
Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 📋 Files Needed

- ✅ `demo/app.py` - Main Streamlit app
- ✅ `demo/requirements.txt` - All dependencies (using `>=`)
- ✅ All source files in `src/`

## 🔗 Full Guides

- **Detailed Guide:** [RAILWAY_DEPLOYMENT.md](./RAILWAY_DEPLOYMENT.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Checklist:** [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

## 💡 Important Notes

- **Free tier:** $5 monthly credit, sleeps after 7 days
- **Wake-up time:** 10-30 seconds (normal for free tier)
- **Auto-deploy:** Push to GitHub = automatic deployment
- **No keep-alive scripts:** Violates hosting terms

## 🐛 Quick Fixes

**App won't start?**
- Check start command has `$PORT` and `0.0.0.0`
- Verify all dependencies in `requirements.txt`

**Build fails?**
- Ensure all packages use `>=` (not `==`)
- Check Railway build logs for specific errors

**App crashes?**
- Check Railway runtime logs
- Verify all imports are in `requirements.txt`

---

**Need more help?** See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

