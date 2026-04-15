@echo off
echo 🚀 Starting AEGIS Earth System
echo ================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

:: Check if model exists, if not train it
if not exist "models\aegis_impact_voter_v01.joblib" (
    echo 📡 Model not found. Training new model...
    python train_and_save.py
) else (
    echo ✅ Model found.
)

:: Start the FastAPI server
echo.
echo 🌐 Starting API server on http://localhost:8000
echo 📝 API Documentation: http://localhost:8000/docs
echo.
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000