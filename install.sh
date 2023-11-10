#!/usr/bin/env bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt --no-cache-dir
echo "Select device"
echo "1. Nvidia GPU"
echo "2. AMD ROCM"
echo "3. CPU"
read -p "Select device: " device
if [[ $device == "1" ]]; then
  pip uninstall -y torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
  python -m venv venvapi
  source venvapi/bin/activate
  pip install -r requirementsapi.txt --no-cache-dir
  ./run.sh
fi
if [[ $device == "2" ]]; then
  pip uninstall -y torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 --no-cache-dir
  python -m venv venvapi
  source venvapi/bin/activate
  pip install -r requirementsapi.txt --no-cache-dir
  ./run.sh
fi
if [[ $device == "3" ]]; then
  python -m venv venvapi
  source venvapi/bin/activate
  pip install -r requirementsapi.txt --no-cache-dir
  ./run.sh
fi
