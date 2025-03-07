@echo off
echo Installing Python 3.10 for SC/ST Act Research Assistant...
echo.
echo This script will download Python 3.10.11 installer.
echo When the installer opens:
echo 1. Check "Add Python 3.10 to PATH" at the bottom of the first screen
echo 2. Click "Install Now" (or customize if you prefer)
echo 3. Wait for installation to complete
echo 4. Click "Close" when finished
echo.
echo Press any key to start downloading...
pause > nul

echo Downloading Python 3.10.11...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile 'python-3.10.11-amd64.exe'"

echo.
echo Download complete. Starting installer...
echo IMPORTANT: Remember to check "Add Python 3.10 to PATH"
echo.
start python-3.10.11-amd64.exe

echo.
echo After installation completes and you've closed the installer:
echo 1. Press any key to continue with setup
echo 2. We'll clean up the installer file
echo 3. Then you can run install_enhanced_search.bat to install the required packages
echo.
pause > nul

echo Cleaning up installer file...
del python-3.10.11-amd64.exe

echo.
echo Python 3.10.11 installation process complete!
echo.
echo Next steps:
echo 1. Open a new Command Prompt (to refresh PATH)
echo 2. Run install_enhanced_search.bat to install the required packages
echo 3. Then run run.bat to start the application
echo.
pause
