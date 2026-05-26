@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
set "ARP_VERSION=0.0.0"
if exist "%ROOT%VERSION" set /p ARP_VERSION=<"%ROOT%VERSION"
set "ARP_COMMIT=unknown"
for /f "usebackq delims=" %%H in (`git -C "%ROOT%" rev-parse --short HEAD 2^>nul`) do set "ARP_COMMIT=%%H"
if "%ARP_COMMIT%"=="unknown" call :read_git_head
if "%ARP_COMMIT%"=="unknown" (
    echo ARP %ARP_VERSION%
) else (
    echo ARP %ARP_VERSION%-%ARP_COMMIT%
)
call "%ROOT%wrappers\_python.bat" -m ai_remaster_gui %*
exit /b %ERRORLEVEL%

:read_git_head
set "HEAD_PATH=%ROOT%.git\HEAD"
if not exist "!HEAD_PATH!" exit /b 0
set /p HEAD_VALUE=<"!HEAD_PATH!"
if "!HEAD_VALUE:~0,5!"=="ref: " (
    set "REF_NAME=!HEAD_VALUE:~5!"
    set "REF_PATH=%ROOT%.git\!REF_NAME:/=\!"
    if exist "!REF_PATH!" set /p HEAD_VALUE=<"!REF_PATH!"
    if not exist "!REF_PATH!" if exist "%ROOT%.git\packed-refs" (
        for /f "usebackq tokens=1,2" %%A in ("%ROOT%.git\packed-refs") do if "%%B"=="!REF_NAME!" set "HEAD_VALUE=%%A"
    )
)
if not "!HEAD_VALUE!"=="" set "ARP_COMMIT=!HEAD_VALUE:~0,7!"
exit /b 0
