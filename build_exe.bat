@echo off
setlocal

echo ==============================================
echo Building Standalone Executable with PyInstaller
echo ==============================================

:: Navigate to the script directory
cd /d "%~dp0"

:: Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements and PyInstaller
echo Installing requirements...
pip install -r btc_bias\requirements.txt
pip install pyinstaller

:: Build the executable
echo Building executable...
cd btc_bias
pyinstaller --onefile --noconsole btc_bias.py

echo ==============================================
echo Build finished! 
echo Your executable is located in btc_bias\dist\btc_bias.exe
echo ==============================================
deactivate
pause
