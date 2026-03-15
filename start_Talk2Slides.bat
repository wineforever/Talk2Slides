call "C:\ProgramData\Miniconda3\Scripts\activate.bat" "C:\Users\mywin\.conda\"
C:
cd "C:\WorkSpace\Talk2Slides"

cd /d %~dp0backend

call venv\Scripts\activate.bat

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause