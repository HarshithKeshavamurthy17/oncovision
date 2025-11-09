# Keeping Streamlit App Always Live

Streamlit Cloud free tier apps can sleep after inactivity. Here are solutions:

## Option 1: Streamlit Cloud Pro (Recommended)
- Upgrade to Streamlit Cloud Pro ($20/month)
- Apps never sleep
- Better performance
- More resources

## Option 2: Uptime Monitoring (Free)
Use a free uptime monitoring service to ping your app every few minutes:

### Services to use:
1. **UptimeRobot** (free, 50 monitors)
   - Go to https://uptimerobot.com
   - Add monitor
   - URL: Your Streamlit app URL
   - Monitoring interval: 5 minutes

2. **cron-job.org** (free)
   - Create account
   - Add cron job
   - URL: Your Streamlit app URL
   - Schedule: Every 5 minutes

3. **Pingdom** (free tier)
   - Similar setup

## Option 3: Alternative Hosting

### Render.com
- Free tier with longer sleep times
- Better for always-on apps

### Railway.app
- Free tier available
- Good for always-on apps

### Heroku
- Paid but reliable
- Always-on option

## Option 4: GitHub Actions (Free)
Create a GitHub Action that pings your app:

```yaml
# .github/workflows/keep-alive.yml
name: Keep App Alive
on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes
  workflow_dispatch:

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Streamlit App
        run: |
          curl -f https://your-app-url.streamlit.app || echo "Ping failed"
```

## Quick Setup (UptimeRobot - Recommended)

1. Go to https://uptimerobot.com
2. Sign up (free)
3. Click "Add New Monitor"
4. Monitor Type: HTTP(s)
5. Friendly Name: OncoVision Demo
6. URL: Your Streamlit app URL
7. Monitoring Interval: 5 minutes
8. Save

Your app will be pinged every 5 minutes, keeping it awake!

## Current App URL
Your app URL: `https://oncovision-akj8dwacntroekz8qxa7gs.streamlit.app`

Add this to UptimeRobot to keep it alive!

