py -m venv venv
call venv\scripts\activate
pip install -r requirements.txt --no-cache-dir
py -m venv venvapi
call venvapi\scripts\activate
pip install -r requirementsapi.txt --no-cache-dir
start run.bat