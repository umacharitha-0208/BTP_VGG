@echo off
echo ============================================================
echo  RIPE Training Launcher
echo  Python: C:\Program Files\Python310\python.exe
echo ============================================================

"C:\Program Files\Python310\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('[OK] CUDA:', torch.cuda.get_device_name(0))"
if %errorlevel% neq 0 (
    echo ERROR: CUDA check failed. Check your PyTorch installation.
    pause
    exit /b 1
)

echo Starting training...
"C:\Program Files\Python310\python.exe" -m ripe.train %*
pause
