# Railway Start Command Fix - "File does not exist: app.py"

## 🐛 Problem
Railway shows error: `Error: Invalid value: File does not exist: app.py`

This happens when Railway can't find the `app.py` file in the working directory.

## ✅ Solution

### Option 1: Verify Root Directory Setting (RECOMMENDED)

1. Go to Railway **Settings** → **Root Directory**
2. Make sure it's set to: `demo` (not `/` or empty)
3. Go to **Settings** → **Deploy** → **Start Command**
4. Set it to:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. **Save** and Railway will redeploy

### Option 2: Use Absolute Path in Start Command

If Option 1 doesn't work, try using the full path:

1. Go to **Settings** → **Deploy** → **Start Command**
2. Set it to:
   ```bash
   streamlit run /app/app.py --server.port $PORT --server.address 0.0.0.0
   ```
   OR if root directory is `/`:
   ```bash
   streamlit run /app/demo/app.py --server.port $PORT --server.address 0.0.0.0
   ```

### Option 3: Check File Structure

Verify that Railway is copying files correctly:

1. Check Railway **Deploy Logs** - look for "copy" steps
2. Make sure `app.py` is being copied to the build directory
3. If using root directory `demo`, Railway should copy `demo/*` to `/app/`

### Option 4: Use Working Directory Command

Try setting the working directory explicitly:

1. **Settings** → **Deploy** → **Start Command**:
   ```bash
   cd /app && streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

## 🔍 Debugging Steps

1. **Check Railway Build Logs:**
   - Look for "copy" steps
   - Verify `app.py` is listed in copied files
   - Check if there are any errors during file copy

2. **Verify Root Directory:**
   - Go to Settings → Root Directory
   - Should be: `demo` (not `/` or empty)
   - If it's `/`, change start command to: `streamlit run demo/app.py ...`

3. **Check File Permissions:**
   - Railway should handle this automatically
   - But if issues persist, check build logs

## 📋 Correct Configuration

**If Root Directory = `demo`:**
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Requirements: `demo/requirements.txt` (auto-detected)

**If Root Directory = `/` (root):**
- Start Command: `streamlit run demo/app.py --server.port $PORT --server.address 0.0.0.0`
- Requirements: `requirements.txt` (in root)

## 🚀 Quick Fix

**Most likely solution:**
1. Go to Railway **Settings**
2. Set **Root Directory** to: `demo`
3. Set **Start Command** to: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. **Save** - Railway will auto-redeploy

---

**The issue:** Railway's working directory doesn't match where `app.py` is located. Setting root directory to `demo` fixes this.

