@echo off
chcp 65001 >nul
echo ========================================
echo   启动语音情感分类服务端 (FastAPI)
echo ========================================
echo.
echo [1/2] 检查是否存在模型文件...
if not exist "models\emotion_classifier.onnx" (
    echo [!] 未找到 models\emotion_classifier.onnx
    echo     将先生成测试模型（随机初始化，仅用于流程验证）
    echo.
    python scripts\generate_dummy_model.py
    if errorlevel 1 (
        echo.
        echo [错误] 模型生成失败，请检查依赖是否已安装
        pause
        exit /b 1
    )
)
echo.
echo [2/2] 启动 FastAPI 服务 (http://localhost:8000)
echo      API文档: http://localhost:8000/docs
echo      按 Ctrl+C 停止服务
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
