@echo off
echo Starting SC/ST Act Research Assistant...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if enhanced search dependencies are installed
python -c "import transformers, torch, sklearn" 2>NUL
if errorlevel 1 (
    echo.
    echo WARNING: Enhanced search dependencies are not installed.
    echo The application will run with limited search capabilities.
    echo.
    echo To enable enhanced search with legal embeddings, run:
    echo     install_enhanced_search.bat
    echo.
    timeout /t 5
)

REM Run the Streamlit app
streamlit run app.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat
