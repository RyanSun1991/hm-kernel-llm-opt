@echo off
REM HMOpt Device Command Relay Service - Windows Launcher
REM Set RELAY_PORT and RELAY_SECRET environment variables before running.

if not defined RELAY_PORT set RELAY_PORT=9100
if not defined RELAY_SECRET set RELAY_SECRET=

echo Starting HMOpt Device Relay on port %RELAY_PORT% ...
python "%~dp0relay_service.py" --port %RELAY_PORT%
pause
