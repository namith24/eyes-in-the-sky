@echo off
rem Eyes in the Sky - one-click launcher for the ground-station laptop.
rem Starts the app (if it is not already running) and opens it in the browser. Works offline.
title Eyes in the Sky
cd /d "%~dp0"
set YOLO_OFFLINE=1
set URL=http://localhost:8501

call :alive
if %errorlevel%==0 goto open

echo Starting Eyes in the Sky...  (keep the small "Eyes in the Sky - server" window open while you use the app)
start "Eyes in the Sky - server" /min python -m streamlit run app\demo_app.py --server.port 8501 --server.headless true
for /l %%i in (1,1,60) do (
    timeout /t 1 /nobreak >nul
    call :alive
    if not errorlevel 1 goto open
)
echo The app did not start within 60 seconds. Open the "Eyes in the Sky - server" window to see why.
pause
exit /b 1

:open
start "" %URL%
exit /b 0

:alive
powershell -NoProfile -Command "try { Invoke-WebRequest %URL%/_stcore/health -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
exit /b %errorlevel%
