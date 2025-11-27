@echo off
echo Starting SmartVisionQA Backend Server...
cd backend
..\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
