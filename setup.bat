@echo off
echo Setting up SC/ST Act Research Assistant...

REM Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or later.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --no-cache-dir

REM Verify streamlit installation
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo Streamlit installation failed. Trying again...
    pip install streamlit==1.31.0 --no-cache-dir
)

REM Verify document processing libraries
python -c "import PyPDF2, docx, pptx" 2>NUL
if errorlevel 1 (
    echo Document processing libraries installation failed. Trying again...
    pip install PyPDF2==3.0.1 python-docx==1.1.0 python-pptx==1.0.1 --no-cache-dir
)

echo.
echo Setup completed successfully!
echo To run the application, use run.bat
echo.
pause
