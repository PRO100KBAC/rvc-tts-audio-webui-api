@echo off
call venv\scripts\activate
start app.py
timeout /t 30 /nobreak
call venvapi\scripts\activate
start api.py