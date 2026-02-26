@echo off
REM URIS 快速启动脚本 (Windows)

echo ========================================
echo 🎬 URIS Video Reasoning Assistant
echo ========================================
echo.

REM 检查是否在正确的目录
if not exist "app.py" (
    echo ❌ 错误: 请在 URIS 项目根目录运行此脚本
    pause
    exit /b 1
)

REM 检查 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到 Python
    echo    请安装 Python 3.8 或更高版本
    pause
    exit /b 1
)

echo ✅ Python 版本:
python --version
echo.

REM 检查依赖
echo 📦 检查依赖...
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  未找到 streamlit，正在安装依赖...
    pip install -r requirements.txt
) else (
    echo ✅ 依赖已安装
)

echo.
echo 📹 摄像头功能提示:
echo    - 首次使用时 Windows 会请求摄像头权限
echo    - 请在弹出的权限请求中选择"允许"
echo.

REM 询问是否运行测试
set /p test="🧪 是否运行功能测试？(y/n) "
if /i "%test%"=="y" (
    echo.
    echo 运行测试...
    python test_features.py
    echo.
    pause
)

REM 启动应用
echo.
echo 🚀 正在启动 URIS...
echo.
echo 提示:
echo   - 应用会在浏览器中自动打开
echo   - 默认地址: http://localhost:8501
echo   - 按 Ctrl+C 停止应用
echo.

streamlit run app.py

pause
