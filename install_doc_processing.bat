@echo off
echo Installing document processing packages...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install document processing packages
pip install PyPDF2==3.0.1 python-docx==1.1.0 python-pptx==1.0.1 beautifulsoup4==4.12.3 requests==2.31.0 --no-cache-dir

REM Verify installation
python -c "import PyPDF2, docx, pptx, bs4, requests" 2>NUL
if errorlevel 1 (
    echo Some packages failed to install. Please check the error messages above.
) else (
    echo All document processing packages installed successfully!
)

echo.
pause
