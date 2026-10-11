@echo off
title DeepSLAM: 6-DoF Visual SLAM & 3D Reconstruction
call conda activate yolo_env 2>nul || call "C:\Users\User\anaconda3\condabin\conda.bat" activate yolo_env
if "%~1"=="" (
    python deep_slam.py latest
) else (
    python deep_slam.py %*
)
pause
