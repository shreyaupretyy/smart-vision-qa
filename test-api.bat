@echo off
echo Testing Backend API...
echo.
echo 1. Testing Health Endpoint:
curl http://localhost:8000/health
echo.
echo.

echo 2. Testing Root Endpoint:
curl http://localhost:8000/
echo.
echo.

echo 3. Testing Video Endpoint (should return 404 or list):
curl http://localhost:8000/api/video/test-id
echo.
echo.

pause
