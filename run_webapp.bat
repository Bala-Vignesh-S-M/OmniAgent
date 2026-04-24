@echo off
echo Starting Mail Agent Web Server...
call .\venv\Scripts\activate.bat
start http://localhost:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
