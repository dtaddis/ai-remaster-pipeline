@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\qwen_colorize_references.py" %*
