@echo off
setlocal
cd /d "%~dp0\.."
if exist "tools\ComfyUI\venv\Scripts\python.exe" (
    cd /d "tools\ComfyUI"
    call "venv\Scripts\activate.bat"
    python main.py %*
    exit /b %errorlevel%
)
if exist "tools\ComfyUI_windows_portable\ComfyUI\main.py" (
    cd /d "tools\ComfyUI_windows_portable\ComfyUI"
    "..\python_embeded\python.exe" main.py %*
    exit /b %errorlevel%
)
echo Could not find ComfyUI.
echo Clone it to tools\ComfyUI, or place the portable package in tools\ComfyUI_windows_portable.
echo You can also start any existing ComfyUI yourself and use --comfy-url with the runner scripts.
exit /b 1
