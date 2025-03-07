@echo off
echo Installing alternative search dependencies without PyTorch...

REM Activate virtual environment
call venv\Scripts\activate.bat 2>NUL
if errorlevel 1 (
    echo Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Install basic dependencies
echo Installing basic dependencies...
pip install google-generativeai==0.8.4 python-dotenv==1.0.0 streamlit==1.31.0 --no-cache-dir

REM Install scikit-learn for TF-IDF search
echo Installing scikit-learn for TF-IDF search...
pip install scikit-learn==1.3.2 --no-cache-dir

REM Install document processing libraries
echo Installing document processing libraries...
pip install PyPDF2==3.0.1 python-docx==1.1.1 python-pptx==1.0.1 beautifulsoup4==4.12.3 requests==2.31.0 pandas==2.1.4 --no-cache-dir

REM Install tabula-py for better PDF extraction
echo Installing tabula-py for better PDF extraction...
pip install tabula-py==2.9.0 --no-cache-dir

REM Verify installation
python -c "import google.generativeai, sklearn, tabula, pandas, PyPDF2, docx, requests" 2>NUL
if errorlevel 1 (
    echo Installation failed. Please check the error messages above.
) else (
    echo Dependencies installed successfully!
)

echo.
echo Note: The application will use TF-IDF search instead of transformer-based search.
echo This is a simpler approach but will still provide reasonable search results.
echo.
pause