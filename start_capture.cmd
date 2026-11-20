@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment not found. Follow First-time setup in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\capture.py
if errorlevel 1 pause
