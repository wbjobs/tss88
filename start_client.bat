@echo off
chcp 65001 >nul
echo ========================================
echo   启动语音情感识别 - 实时模式 (CLI)
echo ========================================
echo.

set SERVER_URL=http://localhost:8000
if not "%1"=="" set SERVER_URL=%1

echo 服务端地址: %SERVER_URL%
echo 按 Ctrl+C 退出
echo.
python client.py --server %SERVER_URL%
pause
