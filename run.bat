@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0tools;%PATH%"
rem Keep Streamlit's home/cache local to this app instead of the Windows profile.
rem Preserve the real Hugging Face cache before redirecting USERPROFILE below.
set "HF_HOME=%USERPROFILE%\.cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "USERPROFILE=%~dp0"
set "STREAMLIT_CONFIG_DIR=%~dp0.streamlit"
if not exist .venv\Scripts\activate.bat (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg was not found on PATH. Install FFmpeg and reopen this window.
  pause
  exit /b 1
)
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
