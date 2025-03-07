@echo off
echo Installing language support packages...

REM Attempt to install language detection and translation libraries
pip install langdetect==1.0.9
pip install deep-translator==1.11.4
pip install googletrans==4.0.0-rc1

echo.
echo Language support installation completed.
echo.
echo If you encounter any issues with translations, please ensure you have an internet connection
echo as the translation services require internet access.
echo.
pause
