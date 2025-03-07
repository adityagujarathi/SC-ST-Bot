@echo off
echo Preparing SC/ST Bot for Streamlit Cloud deployment...

REM Initialize git repository if not already initialized
if not exist .git (
    git init
    echo Git repository initialized
)

REM Add all files except those in .gitignore
git add .

REM Commit changes
git commit -m "Prepare for Streamlit Cloud deployment"

REM Add GitHub remote (you'll need to replace with your actual GitHub URL)
echo.
echo Please enter your GitHub username:
set /p github_username=

REM Create the remote URL
set remote_url=https://github.com/%github_username%/SC_ST_Bot.git

REM Add the remote and push
git remote add origin %remote_url%
git branch -M main
git push -u origin main

echo.
echo Code pushed to GitHub!
echo.
echo Next steps:
echo 1. Go to https://streamlit.io/cloud
echo 2. Sign in with your GitHub account
echo 3. Select the SC_ST_Bot repository
echo 4. Add your GEMINI_API_KEY as a secret
echo 5. Deploy the app
echo.
echo Once deployed, you'll get a URL you can share with anyone!
pause
