@echo off
REM Run the desktop app from source (needs Python). Others can use the .exe from GitHub Releases.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv venv
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
)
venv\Scripts\python.exe desktop_app.py
