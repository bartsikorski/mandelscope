@echo off
rem Double-click me.  Sets up Python + all dependencies (first time: a few minutes)
rem and puts a Mandelscope icon on the Desktop and in the Start menu.
setlocal
cd /d "%~dp0"
echo.
echo  === Mandelscope setup ===
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 goto failed
echo.
echo Creating shortcuts ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcuts.ps1"
if errorlevel 1 goto failed
echo.
echo  Done.  Start Mandelscope from the Desktop icon or the Start menu.
echo.
choice /c YN /m "Start Mandelscope now"
if errorlevel 2 exit /b 0
start "" "%~dp0Mandelscope.lnk"
exit /b 0
:failed
echo.
echo  Setup failed - see the messages above.
pause
exit /b 1
