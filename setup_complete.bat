@echo off
echo SC/ST Act Research Assistant - Complete Setup
echo =============================================
echo This script will set up everything needed for the application:
echo 1. Install Python 3.10 (if needed)
echo 2. Create a virtual environment
echo 3. Install all dependencies including enhanced search
echo 4. Set up the environment file
echo.

REM Check if Python 3.10 is already installed
python --version 2>NUL | findstr "3.10" >NUL
if %errorlevel% equ 0 (
    echo Python 3.10 is already installed.
) else (
    echo Downloading Python 3.10.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-3.10.11-amd64.exe'"
    
    echo Installing Python 3.10.11...
    echo Please follow the installation wizard. Make sure to check "Add Python to PATH"
    python-3.10.11-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    
    echo Waiting for installation to complete...
    timeout /t 30
    
    echo Cleaning up...
    del python-3.10.11-amd64.exe
)

REM Create a new virtual environment with Python 3.10
echo Creating a new virtual environment with Python 3.10...
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install all dependencies
echo Installing all dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist .env (
    echo Creating .env file...
    echo Please enter your Gemini API key:
    set /p GEMINI_API_KEY=
    echo GEMINI_API_KEY=%GEMINI_API_KEY% > .env
    echo .env file created with your API key.
) else (
    echo .env file already exists.
)

echo.
echo Setup complete! You can now run the application with run.bat
echo.
pause
