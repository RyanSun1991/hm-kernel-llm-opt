@echo off
REM HMOpt Device Command Relay Service - Windows Launcher
REM Set RELAY_PORT and RELAY_SECRET environment variables before running.

if not defined RELAY_PORT set RELAY_PORT=9100
if not defined RELAY_SECRET set RELAY_SECRET=

REM ---------------------------------------------------------------------------
REM Disable QuickEdit Mode for THIS console window only.
REM
REM QuickEdit Mode (the default on Windows cmd.exe) freezes the parent
REM process's stdout the moment the operator clicks anywhere in the console
REM window — the console enters "Mark / Selection mode" and any subsequent
REM print() call from Python blocks indefinitely until the operator presses
REM Esc or Enter.  Under the legacy single-threaded HTTPServer this also
REM stalls the serve_forever() loop, so no new requests are accepted; under
REM ThreadingHTTPServer only the worker thread blocks but logs still freeze.
REM Either way: confusing.  Disable it for this window so the relay stays
REM responsive even if the operator clicks the console by accident.
REM
REM This is a window-local override (no registry change).  The %~dpnx0 dance
REM is what survives `pause` / `start` differences between Windows builds.
REM ---------------------------------------------------------------------------
for /f "tokens=2 delims==" %%I in ('"wmic process where ProcessId=%%PID%% get ParentProcessId /value 2^>nul"') do set _PARENT_PID=%%I
REM Best-effort: tell THIS shell to drop QuickEdit + Insert / Mark.  Failures
REM (older Windows, restricted policy) are non-fatal — we still redirect
REM stdout below as the real defense.
powershell -NoProfile -Command "$signature='[DllImport(\"kernel32.dll\")]public static extern IntPtr GetStdHandle(int nStdHandle);[DllImport(\"kernel32.dll\")]public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);[DllImport(\"kernel32.dll\")]public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);';$t=Add-Type -MemberDefinition $signature -Name 'C' -Namespace 'W' -PassThru;$h=$t::GetStdHandle(-10);$m=0;[void]$t::GetConsoleMode([ref]$h, [ref]$m); [void]$t::SetConsoleMode($h, ($m -band -bnot 0x40) -band -bnot 0x10)" 2>nul

REM ---------------------------------------------------------------------------
REM Log file: rotate by date so a bad day's stack traces don't grow forever.
REM Live tail from another window:    powershell Get-Content -Wait %LOG_DIR%\relay_<date>.log
REM ---------------------------------------------------------------------------
set LOG_DIR=%~dp0logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f "tokens=2 delims==" %%I in ('"wmic os get LocalDateTime /value 2^>nul"') do set _NOW=%%I
set LOG_FILE=%LOG_DIR%\relay_%_NOW:~0,8%.log

echo Starting HMOpt Device Relay on port %RELAY_PORT% ...
echo Logging to: %LOG_FILE%
echo To follow:  powershell Get-Content -Wait "%LOG_FILE%"
echo.

REM Redirect stdout/stderr to the log file AND keep an unbuffered copy on the
REM console.  The `python -u` makes Python line-buffer stdout so log lines
REM appear promptly instead of being held in the C runtime buffer.
REM
REM We deliberately do NOT pipe through `tee` (not in stock Windows) — instead
REM we just redirect.  If the operator wants live console output, they tail
REM the log file from a SEPARATE window (so this window's QuickEdit state
REM cannot affect the relay anymore).
python -u "%~dp0relay_service.py" --port %RELAY_PORT% >> "%LOG_FILE%" 2>&1

echo.
echo Relay exited.  Last 50 lines of %LOG_FILE%:
echo ---------------------------------------------------------------------------
powershell -NoProfile -Command "Get-Content -Tail 50 '%LOG_FILE%'"
echo ---------------------------------------------------------------------------
pause
