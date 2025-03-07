@echo off
echo Creating desktop application for SC/ST Bot...

REM Install PyInstaller
pip install pyinstaller

REM Create a wrapper script
echo import os > run_app.py
echo import subprocess >> run_app.py
echo subprocess.run(["streamlit", "run", "app.py"]) >> run_app.py

REM Create the executable
pyinstaller --name "SC_ST_Bot" --onefile --windowed run_app.py

echo Desktop application created! You can find it in the dist folder.
pause
