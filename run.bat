@echo off
rem Mandelscope launcher.  Usage:  run.bat            (explorer)
rem                             run.bat render mandelscope\locations\spiral_1e-100.json -o out.png
rem                             run.bat video  my_sequence.seq.json -o zoom.mp4 --size 3840x2160
cd /d "%~dp0"
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python -m mandelscope %*
) else (
    .venv\Scripts\python.exe -m mandelscope %*
)
