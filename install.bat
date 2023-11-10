@echo off
echo "Select device"
echo "1. Nvidia GPU"
echo "2. CPU"
Set /p device=""
if not defined device goto m1
if "%device%"=="1" (
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)
if "%device%"=="2" (
pip install torch torchvision torchaudio --no-cache-dir
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)