@echo off
set "PIPELINE_PYTHON="
if exist "%~dp0..\tools\ComfyUI\venv\Scripts\python.exe" set "PIPELINE_PYTHON=%~dp0..\tools\ComfyUI\venv\Scripts\python.exe"
if not defined PIPELINE_PYTHON if exist "%~dp0..\tools\ComfyUI_windows_portable\python_embeded\python.exe" set "PIPELINE_PYTHON=%~dp0..\tools\ComfyUI_windows_portable\python_embeded\python.exe"
if not defined PIPELINE_PYTHON set "PIPELINE_PYTHON=python"
%PIPELINE_PYTHON% --version >nul 2>nul
if errorlevel 1 (
    echo Could not find Python.
    echo Install Python, or put ComfyUI in tools\ComfyUI with a venv, or set up ComfyUI_windows_portable under tools.
    exit /b 1
)
exit /b 0
