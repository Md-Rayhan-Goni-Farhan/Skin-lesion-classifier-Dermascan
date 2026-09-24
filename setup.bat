@echo off
title DermaScan - Environment Setup
color 0A

echo ============================================================
echo  DermaScan - Development Environment Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 from python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo [3/5] Installing training dependencies...
pip install tensorflow-cpu==2.18.0 scikit-learn Pillow requests --quiet
if errorlevel 1 (
    echo [ERROR] Dependency install failed. Check internet connection.
    pause
    exit /b 1
)

echo.
echo [4/5] Installing GUI + build dependencies...
pip install Pillow pyinstaller --quiet

echo.
echo [5/5] Done!
echo.
echo ============================================================
echo  Next steps:
echo    1. cd model_training
echo    2. python prepare_dataset.py     (downloads ISIC data)
echo    3. python train.py               (trains model ~30-60 min)
echo    4. cd ..
echo    5. build.bat                     (creates EXE)
echo ============================================================
echo.
pause
