@echo off
echo 启动植物病害数据集制作工具...
python start.py
if %errorlevel% neq 0 (
    echo 程序启动失败，请检查错误信息。
    pause
    exit /b %errorlevel%
)
pause 