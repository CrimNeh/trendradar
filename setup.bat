@echo off
REM Double-click this file. No PowerShell policy, no encoding issues.
REM It does the same thing as scripts\setup.ps1.
setlocal

cd /d "%~dp0"

echo.
echo === TrendRadar setup ===
echo.

if not exist "requirements.txt" (
    echo ERROR: wrong folder.
    echo This file must sit in the folder that contains README.md and src.
    echo Current folder: %CD%
    echo.
    pause
    exit /b 1
)

echo --- Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install it from python.org and tick "Add Python to PATH".
    echo Then close this window and run setup again.
    echo.
    pause
    exit /b 1
)
python --version

echo.
echo --- Virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create .venv
        pause
        exit /b 1
    )
    echo created .venv
) else (
    echo .venv already exists
)

echo.
echo --- Dependencies: 2-4 minutes, please wait ---
".venv\Scripts\pip.exe" install --upgrade pip --quiet
".venv\Scripts\pip.exe" install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo ERROR: could not install dependencies.
    echo Read the message above. Check internet, antivirus, disk space.
    pause
    exit /b 1
)
echo dependencies installed

echo.
echo --- Database ---
".venv\Scripts\python.exe" src\db.py
if errorlevel 1 (
    echo ERROR: could not create the database.
    pause
    exit /b 1
)

echo.
echo === Ready ===
echo.
echo Next, run these from this same folder:
echo.
echo     .venv\Scripts\python.exe src\collect.py
echo     .venv\Scripts\python.exe src\youtube_collect.py --seed
echo     .venv\Scripts\python.exe src\origins.py
echo     .venv\Scripts\python.exe src\report.py
echo     .venv\Scripts\python.exe -m streamlit run dashboard\app.py
echo.
pause
