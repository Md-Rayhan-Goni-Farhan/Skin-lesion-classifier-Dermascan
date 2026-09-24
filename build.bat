@echo off
title DermaScan - Building EXE
color 0A

echo ============================================================
echo  DermaScan - Building Standalone EXE
echo ============================================================
echo.

call venv\Scripts\activate.bat

:: Verify model exists
if not exist "model_training\model_weights.keras" (
    echo [ERROR] model_weights.keras not found in model_training\
    echo         Run train.py first.
    pause
    exit /b 1
)

if not exist "model_training\model_meta.json" (
    echo [ERROR] model_meta.json not found in model_training\
    pause
    exit /b 1
)

echo [1/3] Copying model files to project root...
copy "model_training\model_weights.keras" "model_weights.keras"
copy "model_training\model_meta.json"  "model_meta.json"

echo.
echo [2/3] Running PyInstaller...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "DermaScan" ^
    --add-data "model_weights.keras;." ^
    --add-data "model_meta.json;." ^
    --add-data "assets;assets" ^
    --hidden-import="tensorflow" ^
    --hidden-import="tensorflow.keras" ^
    --hidden-import="PIL._tkinter_finder" ^
    --collect-all tensorflow ^
    --collect-all keras ^
    src\app.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. See above for details.
    pause
    exit /b 1
)

echo.
echo [3/3] Creating release zip...
if not exist "release" mkdir release
copy "dist\DermaScan.exe" "release\DermaScan.exe"
copy "README.md" "release\README.md"

:: Create zip using PowerShell
powershell -Command "Compress-Archive -Path 'release\*' -DestinationPath 'DermaScan_v1.0_Windows.zip' -Force"

echo.
echo ============================================================
echo  Build complete!
echo  EXE:  dist\DermaScan.exe
echo  ZIP:  DermaScan_v1.0_Windows.zip  (upload this to GitHub)
echo ============================================================
echo.
pause
