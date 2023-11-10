@echo off
py -m venv venv
call venv\scripts\activate
pip install -r requirements.txt --no-cache-dir
echo "Select device"
echo "1. Nvidia GPU cu118"
echo "2. Nvidia GPU cu121"
echo "3. CPU"
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
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)
if "%device%"=="3" (
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat
)
