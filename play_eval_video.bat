@echo off
if exist "debug\20261009_154542\da2_geometry_eval.mp4" (
    start "" "debug\20261009_154542\da2_geometry_eval.mp4"
) else (
    echo Evaluation video not found yet. Run run_da2_test.bat first!
)
