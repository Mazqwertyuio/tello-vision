@echo off
call conda activate yolo_env 2>nul || call "C:\Users\User\anaconda3\condabin\conda.bat" activate yolo_env
if "%~1"=="" (
    python test_da2_flight_offline.py --flight latest
) else (
    python test_da2_flight_offline.py %*
)
pause
