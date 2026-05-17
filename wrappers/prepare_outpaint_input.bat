@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\prepare_outpaint_input.py" %*
