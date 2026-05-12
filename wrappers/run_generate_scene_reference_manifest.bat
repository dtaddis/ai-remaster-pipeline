@echo off
setlocal
cd /d "%~dp0\.."
call wrappers\_python.bat
if errorlevel 1 exit /b %errorlevel%
if exist "C:\Program Files\ffmpeg\bin\ffmpeg.exe" set "PATH=C:\Program Files\ffmpeg\bin;%PATH%"
"%PIPELINE_PYTHON%" scripts\generate_scene_reference_manifest.py %*
