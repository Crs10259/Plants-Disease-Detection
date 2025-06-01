@echo off
echo 安装植物病害数据集制作工具所需依赖...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 安装依赖失败，请检查错误信息。
    pause
    exit /b %errorlevel%
)
echo 依赖安装成功！现在可以运行 run_tool.bat 启动工具。
pause 