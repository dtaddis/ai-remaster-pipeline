@echo off
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%PY%" goto run
set "PY=py"
:run
"%PY%" %*
exit /b %ERRORLEVEL%
