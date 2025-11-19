@echo off
REM Activate environment
call ..\venv\Scripts\activate.bat

REM Run the pipeline from project root
cd ..
python run_pipeline.py
