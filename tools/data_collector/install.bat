@echo off
echo install data collector dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo install dependencies failed, please check the error information.
    pause
    exit /b %errorlevel%
)
echo dependencies installed successfully! now you can run run_tool.bat to start the tool.
pause 