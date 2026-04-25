@echo off
setlocal

echo ==============================================
echo Bitcoin Bias Predictor - Setup and Run Script
echo ==============================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python to run this script.
    pause
    exit /b 1
)

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

:: Install requirements
if exist "btc_bias\requirements.txt" (
    echo Installing requirements...
    pip install -r btc_bias\requirements.txt
) else (
    echo requirements.txt not found. Skipping dependency installation.
)

:: Run the script
echo.
echo Running btc_bias.py...
echo ==============================================
python btc_bias\btc_bias.py

:: Deactivate and pause when done
echo ==============================================
echo Execution finished.
deactivate
pause
