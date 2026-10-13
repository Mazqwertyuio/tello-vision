@echo off
title Tello Visual LiDAR Autopilot
call conda activate yolo_env 2>nul || call "C:\Users\User\anaconda3\condabin\conda.bat" activate yolo_env 2>nul
python main.py %*
pause
