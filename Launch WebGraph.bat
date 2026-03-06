@echo off
title WebGraph Control Panel
cd /d "%~dp0"

:: Try to find Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Launch the GUI (pythonw hides the extra console window)
where pythonw >nul 2>&1
if %errorlevel% equ 0 (
    start "" pythonw gui.py
) else (
    python gui.py
)
