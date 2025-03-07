@echo off
echo Stopping any running processes...
taskkill /F /IM python.exe /T

echo Cleaning up old virtual environment...
rmdir /S /Q venv

echo Running setup...
call setup.bat

echo.
echo Setup complete! Now you can run the application with run.bat
pause
