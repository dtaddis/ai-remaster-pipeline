@echo off
setlocal
call "%~dp0_python.bat" "%~dp0..\scripts\generate_references.py" %*
