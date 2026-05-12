@echo off
setlocal
cd /d "%~dp0\.."
call wrappers\_python.bat
if errorlevel 1 exit /b %errorlevel%
"%PIPELINE_PYTHON%" scripts\generate_transnet_reference_manifest.py %*
