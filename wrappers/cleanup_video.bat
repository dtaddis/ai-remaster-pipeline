@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\cleanup_video.py" %*
