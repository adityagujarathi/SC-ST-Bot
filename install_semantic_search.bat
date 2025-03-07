@echo off
echo Installing semantic search dependencies...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install sentence-transformers, FAISS, and scikit-learn
pip install sentence-transformers==2.2.2 faiss-cpu==1.7.4 scikit-learn==1.3.2 --no-cache-dir

REM Verify installation
python -c "import sentence_transformers, faiss, sklearn" 2>NUL
if errorlevel 1 (
    echo Installation failed. Please check the error messages above.
) else (
    echo Semantic search dependencies installed successfully!
)

echo.
echo Note: The first time you run the application, it will download the legal embedding model.
echo This may take a few minutes depending on your internet connection.
echo.
pause
