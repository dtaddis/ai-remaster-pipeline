@echo off
setlocal
set "ROOT=%~dp0"
set "ARP_VERSION=0.0.0"
if exist "%ROOT%VERSION" set /p ARP_VERSION=<"%ROOT%VERSION"
set "ARP_COMMIT=unknown"
for /f "usebackq delims=" %%H in (`git -C "%ROOT%" rev-parse --short HEAD 2^>nul`) do set "ARP_COMMIT=%%H"
echo ARP %ARP_VERSION%
echo Commit %ARP_COMMIT%
call "%ROOT%wrappers\_python.bat" -m ai_remaster_gui %*
exit /b %ERRORLEVEL%
