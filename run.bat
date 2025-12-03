@echo off
REM run.bat - Run Python with MKL environment
REM Usage: run.bat script.py [args...]

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
python %*
