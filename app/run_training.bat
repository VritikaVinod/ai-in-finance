@echo off
python -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1
python train_compare.py
pause
