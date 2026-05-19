@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\generate_single_reference.py" %*
