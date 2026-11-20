@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment not found. Follow docs\CAPTURE.md to set it up.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\capture.py
if errorlevel 1 pause
