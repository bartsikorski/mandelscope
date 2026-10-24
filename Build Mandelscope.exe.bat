@echo off
rem Builds a stand-alone program folder:  dist\Mandelscope\Mandelscope.exe
rem (copy the whole dist\Mandelscope folder anywhere - no Python needed there).
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    ) else (
        echo uv not found - run "Install Mandelscope.bat" first.
        pause
        exit /b 1
    )
)
echo Building Mandelscope.exe (takes 1-3 minutes) ...
uv run --with "pyinstaller>=6.10" pyinstaller packaging\mandelscope.spec --noconfirm --clean --distpath dist --workpath build
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
echo.
echo Built: %~dp0dist\Mandelscope\Mandelscope.exe
explorer "%~dp0dist\Mandelscope"
