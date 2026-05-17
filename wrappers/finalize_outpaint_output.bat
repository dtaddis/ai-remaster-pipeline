@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\finalize_outpaint_output.py" %*
