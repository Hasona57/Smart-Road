@echo off
REM Quick start script for the detection server

echo ========================================
echo Starting Toy Car Detection Server
echo ========================================
echo.

cd /d "%~dp0"
python server.py

pause


