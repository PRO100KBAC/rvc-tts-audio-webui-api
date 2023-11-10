#!/usr/bin/env bash

source venv/bin/activate
python rvc.py
timeout 0s sleep 30s
source venvapi/bin/activate
python api.py
exit
