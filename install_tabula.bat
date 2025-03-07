@echo off
echo Installing tabula-py for better table extraction...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install tabula-py and pandas
pip install tabula-py==2.9.0 pandas==2.1.4 --no-cache-dir

REM Verify installation
python -c "import tabula, pandas" 2>NUL
if errorlevel 1 (
    echo Installation failed. Please check the error messages above.
) else (
    echo Tabula-py and pandas installed successfully!
)

echo.
echo Note: Tabula-py requires Java to be installed on your system.
echo If you don't have Java installed, please download and install it from:
echo https://www.java.com/en/download/
echo.
pause
