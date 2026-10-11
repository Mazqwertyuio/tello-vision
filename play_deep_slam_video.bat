@echo off
set FLIGHT_DIR=debug\20261009_154542
if not "%~1"=="" set FLIGHT_DIR=debug\%~1
if exist "%FLIGHT_DIR%\deep_slam_reconstruction.mp4" (
    start "" "%FLIGHT_DIR%\deep_slam_reconstruction.mp4"
) else (
    echo [ERROR] Reconstruction video not found in %FLIGHT_DIR%
    pause
)
