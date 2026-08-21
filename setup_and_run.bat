@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Python 3 was not found. Install Python 3.10-3.13 and try again.
        pause
        exit /b 1
    )
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check the error above.
    pause
    exit /b 1
)
python -c "import cv2,numpy,pandas,streamlit,ultralytics; print('All POC dependencies are installed.')"
if errorlevel 1 (
    echo.
    echo Dependency verification failed.
    pause
    exit /b 1
)
streamlit run app.py
pause
