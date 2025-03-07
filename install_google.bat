@echo off
echo Installing Google Generative AI package...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install Google Generative AI package
pip install google-generativeai==0.8.4 --no-cache-dir

REM Verify installation
pip show google-generativeai

echo.
echo Installation complete!
pause
