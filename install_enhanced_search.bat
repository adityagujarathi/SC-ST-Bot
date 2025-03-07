@echo off
echo Installing enhanced search dependencies with latest PyTorch...

REM Activate virtual environment
call venv\Scripts\activate.bat 2>NUL
if errorlevel 1 (
    echo Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Install transformers and the latest available PyTorch version
echo Installing transformers and PyTorch 2.6.0...
pip install transformers==4.35.0 torch==2.6.0 numpy==1.24.3 scikit-learn==1.3.2 --no-cache-dir

REM Verify installation
python -c "import transformers, torch, numpy, sklearn" 2>NUL
if errorlevel 1 (
    echo Installation failed. Please check the error messages above.
) else (
    echo Enhanced search dependencies installed successfully!
)

echo.
echo Note: The first time you run the application, it will download the legal model.
echo This may take a few minutes depending on your internet connection.
echo.
pause
