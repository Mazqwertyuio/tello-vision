@echo off
call conda activate yolo_env 2>nul || call "C:\Users\User\anaconda3\condabin\conda.bat" activate yolo_env
if "%~1"=="" (
    python replay_flight.py --da2
) else (
    python replay_flight.py %* --da2
)
pause
