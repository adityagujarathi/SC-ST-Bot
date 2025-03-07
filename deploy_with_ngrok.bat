@echo off
echo Starting SC/ST Bot with ngrok...

REM Start Streamlit in the background
start cmd /k "python -m streamlit run app.py"

REM Wait for Streamlit to start
timeout /t 5

REM Start ngrok to expose the Streamlit port
ngrok http 8501

echo Deployment complete! Share the ngrok URL with others.
echo Press Ctrl+C to stop ngrok when finished.
