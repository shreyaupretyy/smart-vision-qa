@echo off
echo =====================================
echo SmartVisionQA - Setup Script
echo =====================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist uploads mkdir uploads
if not exist temp mkdir temp
if not exist chroma_db mkdir chroma_db

REM Copy environment file
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
)

REM Setup frontend
echo Setting up frontend...
cd frontend
call npm install
cd ..

echo =====================================
echo Setup complete!
echo =====================================
echo To start the backend:
echo   venv\Scripts\activate
echo   uvicorn backend.main:app --reload
echo.
echo To start the frontend:
echo   cd frontend
echo   npm run dev
echo =====================================
pause
