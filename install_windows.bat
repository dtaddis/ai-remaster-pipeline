@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
exit /b %ERRORLEVEL%
