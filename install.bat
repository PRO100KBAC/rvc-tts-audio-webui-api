@echo off
py -m venv venv
call venv\scripts\activate
pip install -r requirements.txt --no-cache-dir
echo "Select device"
echo "1. Nvidia GPU"
echo "2. CPU"
Set /p device=""
if "%device%"=="1" (
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)
if "%device%"=="2" (
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)
