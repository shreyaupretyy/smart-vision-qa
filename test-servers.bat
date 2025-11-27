@echo off
echo Testing SmartVisionQA Servers...
echo.

echo Testing Backend (Port 8000)...
curl -s http://localhost:8000/health
echo.
echo.

echo Testing Frontend (Port 5173)...
curl -s http://localhost:5173
echo.
echo.

echo If you see responses above, servers are working!
echo.
echo Backend should show: {"status":"healthy"}
echo Frontend should show HTML content
echo.
pause
