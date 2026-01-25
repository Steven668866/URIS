@echo off
REM Quick start script for Windows
REM Generates synthetic dataset for Qwen2-VL fine-tuning

echo ========================================
echo 🤖 Qwen2-VL Dataset Generator
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo ❌ .env file not found!
    echo.
    echo Please create .env file from env.example:
    echo   1. Copy env.example to .env
    echo   2. Fill in your OPENAI_API_KEY
    echo   3. Configure BASE_URL and MODEL
    echo.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if dependencies are installed
echo 📦 Checking dependencies...
pip show openai >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dependencies not installed. Installing now...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo ✅ Dependencies installed
)

echo.
echo 🚀 Starting dataset generation...
echo.

REM Run the generator
python generate_data.py

if errorlevel 1 (
    echo.
    echo ❌ Generation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🎉 Generation Complete!
echo ========================================
echo.

REM Validate the dataset
if exist dataset_personalization.json (
    echo 🔍 Validating dataset...
    echo.
    python validate_dataset.py
    echo.
)

echo.
echo 📁 Files generated:
dir /B dataset_*.json 2>nul
echo.

pause






