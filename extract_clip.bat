@echo off
chcp 65001 >nul
echo ========================================
echo 视频片段提取工具 (快速版)
echo ========================================
echo.

if "%1"=="" (
    echo 使用方法:
    echo   extract_clip.bat ^<视频文件^> [开始时间秒] [时长秒]
    echo.
    echo 示例:
    echo   extract_clip.bat video.mp4 0 20
    echo   extract_clip.bat video.mp4 10 30
    echo.
    pause
    exit /b
)

set INPUT_VIDEO=%1
set START_TIME=%2
set DURATION=%3

if "%START_TIME%"=="" set START_TIME=0
if "%DURATION%"=="" set DURATION=20

echo 输入视频: %INPUT_VIDEO%
echo 开始时间: %START_TIME% 秒
echo 片段时长: %DURATION% 秒
echo.

for %%F in ("%INPUT_VIDEO%") do (
    set "OUTPUT=%%~dpnF_clip_%START_TIME%s_%DURATION%s%%~xF"
)

echo 输出文件: %OUTPUT%
echo.

ffmpeg -i "%INPUT_VIDEO%" -ss %START_TIME% -t %DURATION% -c:v libx264 -c:a aac -preset fast -y "%OUTPUT%"

if %ERRORLEVEL%==0 (
    echo.
    echo [成功] 视频片段已提取完成！
    echo 输出文件: %OUTPUT%
) else (
    echo.
    echo [错误] 提取失败，请检查 ffmpeg 是否已安装
)

pause


