@echo off
call venv\scripts\activate
start rvc.py
timeout /t 30 /nobreak
call venvapi\scripts\activate
start api.py
EXIT