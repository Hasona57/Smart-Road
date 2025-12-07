@echo off
REM Setup script for Toy Car Detection Project
REM Run this to prepare your Python environment

echo ========================================
echo Toy Car Detection - Environment Setup
echo ========================================
echo.

REM Check if Python is available (try both 'python' and 'py')
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    echo Python found - using 'python' command
    goto :python_found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    echo Python found - using 'py' command
    goto :python_found
)

echo Python not found!
echo.
echo Please install Python 3.11+ from:
echo https://www.python.org/downloads/windows/
echo.
echo IMPORTANT: Check "Add Python to PATH" during installation!
echo.
pause
exit /b 1

:python_found

echo.

REM Upgrade pip (continue even if it fails - pip might already be up to date)
echo Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip
REM Don't check errorlevel - pip upgrade warnings are not critical
echo.

REM Install PyTorch with CUDA support (for RTX 4060)
echo Installing PyTorch with CUDA 12.1 support...
echo This may take several minutes...
%PYTHON_CMD% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install PyTorch!
    echo Please check the error messages above.
    pause
    exit /b 1
)
echo.

REM Install other dependencies
echo Installing project dependencies...
cd /d "%~dp0\.."
%PYTHON_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)
echo.

REM Test YOLO installation
echo Testing YOLO installation...
%PYTHON_CMD% -c "from ultralytics import YOLO; print('YOLO installed successfully!')"
if errorlevel 1 (
    echo.
    echo ERROR: YOLO installation test failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Collect images of toy cars (police, ambulance, normal)
echo 2. Label them using Roboflow (https://roboflow.com)
echo 3. Export as YOLOv8 format
echo 4. Place in dataset/train/ and dataset/valid/
echo 5. Run: %PYTHON_CMD% toy_car_detection/train.py
echo.
pause
