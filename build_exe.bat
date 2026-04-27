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
pip install -r requirements.txt
pip install pyinstaller

:: Build the executable
echo Building executable...
pyinstaller --onefile --noconsole src\main.py

echo ==============================================
echo Build finished! 
echo Your executable is located in dist\main.exe
echo ==============================================
deactivate
pause
