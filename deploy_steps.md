# Step-by-Step Deployment Guide

## 1. Configure Git (One-time setup)

```
git config --global user.email "your.email@example.com"
git config --global user.name "Your Name"
```

## 2. Initialize Git Repository

```
git init
```

## 3. Add Files to Git

```
git add .
```

## 4. Commit Changes

```
git commit -m "Initial commit for Streamlit Cloud deployment"
```

## 5. Create GitHub Repository

1. Go to https://github.com/new
2. Name the repository: `SC_ST_Bot`
3. Make it Public
4. Click "Create repository"
5. Copy the repository URL (e.g., `https://github.com/yourusername/SC_ST_Bot.git`)

## 6. Link and Push to GitHub

```
git remote add origin https://github.com/yourusername/SC_ST_Bot.git
git branch -M main
git push -u origin main
```

## 7. Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `yourusername/SC_ST_Bot`
5. Set Main file path: `app.py`
6. Click "Advanced settings"
7. Add your Gemini API key under "Secrets":
   ```
   GEMINI_API_KEY = "your-actual-api-key"
   ```
8. Click "Deploy!"

## 8. Share Your App

Once deployed, you'll get a URL like: `https://yourusername-sc-st-bot-app-xxxxx.streamlit.app/`

Share this URL with anyone who needs access to the application!
