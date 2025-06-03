@echo off
echo start data checker...
python start.py
if %errorlevel% neq 0 (
    echo program start failed, please check the error information.
    pause
    exit /b %errorlevel%
)
pause 