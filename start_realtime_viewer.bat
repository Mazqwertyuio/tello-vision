@echo off
call conda activate yolo_env 2>nul || call "C:\Users\User\anaconda3\condabin\conda.bat" activate yolo_env
if "%~1"=="" (
    python live_da2_navigator.py latest
) else (
    python live_da2_navigator.py %*
)
pause
