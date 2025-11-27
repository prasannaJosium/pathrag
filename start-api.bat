@echo off
REM Start script for PathRAG API only

echo Starting PathRAG API...

REM Install dependencies using Poetry
echo Installing backend dependencies...
call poetry install
echo Backend dependencies installed.

REM Start backend API
echo Starting backend API on port 8000...
call poetry run uvicorn main:app --host 0.0.0.0 --port 8000

echo API server stopped.
