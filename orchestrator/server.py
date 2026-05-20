"""Tiny work-queue server.

Holds a queue of (seed, chain) jobs and accepts probe-result posts back. Designed
for: phones run this and an aggregator; Mac mini runs workers that pull and
execute. Stdlib only — no Flask, no Redis, no Celery. Persistence is filesystem
JSONL; if the process dies, in-flight jobs revert to pending on next start.

Endpoints:
    GET  /jobs/next        -> {"job_id": ..., "behavior": ..., "chain": [...]} or 204
    POST /jobs/{id}/result -> body is probe JSON; server saves and marks done
    GET  /status           -> counts of pending/in_flight/done
    POST /jobs/submit      -> body: {"behavior": ..., "chain": [...]} -> {"job_id": ...}

State files (in --state-dir, default ./orchestrator/state):
    pending.jsonl   one job per line
    in_flight.json  {job_id: {claimed_at, claimed_by}}
    done/<id>.json  finished probe result
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.RLock()


class Queue:
    def __init__(self, state_dir: str, lease_seconds: int = 600):
        self.state_dir = state_dir
        self.done_dir = os.path.join(state_dir, "done")
        self.pending_path = os.path.join(state_dir, "pending.jsonl")
        self.in_flight_path = os.path.join(state_dir, "in_flight.json")
        self.lease_seconds = lease_seconds
        os.makedirs(self.done_dir, exist_ok=True)
        if not os.path.exists(self.pending_path):
            open(self.pending_path, "a").close()
        if not os.path.exists(self.in_flight_path):
            with open(self.in_flight_path, "w") as f:
                json.dump({}, f)

    def _read_pending(self) -> list[dict]:
        with open(self.pending_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write_pending(self, jobs: list[dict]) -> None:
        with open(self.pending_path, "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")

    def _read_in_flight(self) -> dict:
        with open(self.in_flight_path) as f:
            return json.load(f)

    def _write_in_flight(self, d: dict) -> None:
        with open(self.in_flight_path, "w") as f:
            json.dump(d, f)

    def _reap_expired(self) -> None:
        in_flight = self._read_in_flight()
        now = time.time()
        expired = [jid for jid, meta in in_flight.items() if now - meta["claimed_at"] > self.lease_seconds]
        if not expired:
            return
        pending = self._read_pending()
        for jid in expired:
            pending.append(in_flight.pop(jid)["job"])
        self._write_pending(pending)
        self._write_in_flight(in_flight)

    def submit(self, behavior: str, chain: list[str], seed_id: str | None = None) -> str:
        with LOCK:
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "seed_id": seed_id or job_id[:8], "behavior": behavior, "chain": chain}
            with open(self.pending_path, "a") as f:
                f.write(json.dumps(job) + "\n")
            return job_id

    def claim(self, worker_id: str) -> dict | None:
        with LOCK:
            self._reap_expired()
            pending = self._read_pending()
            if not pending:
                return None
            job = pending.pop(0)
            self._write_pending(pending)
            in_flight = self._read_in_flight()
            in_flight[job["job_id"]] = {"claimed_at": time.time(), "claimed_by": worker_id, "job": job}
            self._write_in_flight(in_flight)
            return job

    def complete(self, job_id: str, probe: dict) -> bool:
        with LOCK:
            in_flight = self._read_in_flight()
            if job_id not in in_flight:
                return False
            del in_flight[job_id]
            self._write_in_flight(in_flight)
            with open(os.path.join(self.done_dir, f"{job_id}.json"), "w") as f:
                json.dump(probe, f, indent=2, default=str)
            return True

    def status(self) -> dict:
        with LOCK:
            self._reap_expired()
            return {
                "pending": len(self._read_pending()),
                "in_flight": len(self._read_in_flight()),
                "done": len([n for n in os.listdir(self.done_dir) if n.endswith(".json")]),
            }


_BOOTSTRAP_TEMPLATE = """#!/usr/bin/env bash
# agent-farm worker bootstrap. Pulls worker.py from {origin}, runs it.
# Re-run with TARGET=anthropic / TARGET=openai if pointing at a remote model
# (worker still pulls jobs from this orchestrator).
set -euo pipefail
ORCH="{origin}"
WORKDIR="${{HOME}}/.agent-farm-worker"
mkdir -p "$WORKDIR/node/jailbreak"
echo "[*] downloading worker code from $ORCH"
curl -fsSL "$ORCH/worker.py" -o "$WORKDIR/worker.py"
curl -fsSL "$ORCH/bundle.tar" -o "$WORKDIR/bundle.tar"
tar -xf "$WORKDIR/bundle.tar" -C "$WORKDIR"
cd "$WORKDIR"
TARGET="${{TARGET:-ollama}}"
MODEL_ARG=""
[ -n "${{MODEL:-}}" ] && MODEL_ARG="--model $MODEL"
echo "[*] starting worker target=$TARGET server=$ORCH"
exec python3 -m worker --server "$ORCH" --target "$TARGET" $MODEL_ARG
"""


def _origin(req) -> str:
    host = req.headers.get("Host") or f"{req.server.server_address[0]}:{req.server.server_address[1]}"
    return f"http://{host}"


def _render_bootstrap(req) -> str:
    return _BOOTSTRAP_TEMPLATE.format(origin=_origin(req))


_BOOTSTRAP_PS1 = r"""# agent-farm worker bootstrap (Windows PowerShell).
$ErrorActionPreference = "Stop"
$ORCH = "{origin}"
$WORKDIR = Join-Path $HOME ".agent-farm-worker"
New-Item -ItemType Directory -Force -Path $WORKDIR | Out-Null
Write-Host "[*] downloading worker code from $ORCH"
Invoke-WebRequest -Uri "$ORCH/worker.py" -OutFile (Join-Path $WORKDIR "worker.py") -UseBasicParsing
Invoke-WebRequest -Uri "$ORCH/bundle.tar" -OutFile (Join-Path $WORKDIR "bundle.tar") -UseBasicParsing
tar -xf (Join-Path $WORKDIR "bundle.tar") -C $WORKDIR
Set-Location $WORKDIR
$target = if ($env:TARGET) {{ $env:TARGET }} else {{ "ollama" }}
$args = @("-m","worker","--server",$ORCH,"--target",$target)
if ($env:MODEL) {{ $args += @("--model",$env:MODEL) }}
Write-Host "[*] starting worker target=$target server=$ORCH"
& python @args
"""


def _render_bootstrap_ps1(req) -> str:
    return _BOOTSTRAP_PS1.format(origin=_origin(req))


_BOOTSTRAP_PY = r'''"""Cross-platform agent-farm worker bootstrap.

Usage:
    curl -o b.py {origin}/bootstrap.py && python b.py
or in PowerShell:
    iwr {origin}/bootstrap.py -OutFile b.py; python b.py
"""
import os, sys, tarfile, urllib.request, subprocess

ORCH = "{origin}"
WORKDIR = os.path.join(os.path.expanduser("~"), ".agent-farm-worker")
os.makedirs(WORKDIR, exist_ok=True)
print(f"[*] downloading worker code from {{ORCH}}")
urllib.request.urlretrieve(f"{{ORCH}}/worker.py", os.path.join(WORKDIR, "worker.py"))
urllib.request.urlretrieve(f"{{ORCH}}/bundle.tar", os.path.join(WORKDIR, "bundle.tar"))
with tarfile.open(os.path.join(WORKDIR, "bundle.tar")) as t:
    t.extractall(WORKDIR)
target = os.environ.get("TARGET", "ollama")
cmd = [sys.executable, "-m", "worker", "--server", ORCH, "--target", target]
if os.environ.get("MODEL"):
    cmd += ["--model", os.environ["MODEL"]]
print(f"[*] starting worker target={{target}} server={{ORCH}}")
os.chdir(WORKDIR)
sys.exit(subprocess.call(cmd))
'''


def _render_bootstrap_py(req) -> str:
    return _BOOTSTRAP_PY.format(origin=_origin(req))


_JOINER_BAT = r"""@echo off
setlocal enabledelayedexpansion
title NAT NETWORK - Node Setup
color 0B

set "ORCH={origin}"
set "WORKDIR=%USERPROFILE%\.nat-network"
set "MODEL=qwen2.5-coder:1.5b"

cls
echo.
echo  ============================================================
echo                       N A T   N E T W O R K
echo                          Node Joiner
echo  ============================================================
echo.
echo   Server: %ORCH%
echo   Workdir: %WORKDIR%
echo.

if exist "%WORKDIR%\.consent" (
    echo  [Consent on file from %WORKDIR%\.consent]
    goto :install
)

echo  Before you join, here is what this does:
echo.
echo   * Installs Python and Ollama on this PC if not already present
echo   * Downloads a local AI model ^(~1 GB^) you can use anytime
echo   * Opens a worker window that runs AI safety evaluation jobs
echo     against your local Ollama, and reports verdicts back to NAT
echo   * Uses CPU / RAM / network while the worker window is open
echo.
echo  How to stop:
echo   * Close the "NAT NETWORK NODE" window any time
echo   * The local AI model stays installed and usable standalone
echo   * To uninstall fully, delete %WORKDIR% and `ollama rm %MODEL%`
echo.
echo  Workload note: jobs include AdvBench safety probes. The local
echo  model only refuses or responds; nothing is exfiltrated beyond
echo  your verdict label going back to the NAT orchestrator.
echo.

choice /c YN /n /m " Type Y to consent and join NAT NETWORK, or N to abort: "
if errorlevel 2 (
    echo Aborted. Nothing installed.
    timeout /t 3 >nul
    exit /b 1
)

if not exist "%WORKDIR%" mkdir "%WORKDIR%"
echo consent_at=%date% %time%> "%WORKDIR%\.consent"
echo orchestrator=%ORCH%>> "%WORKDIR%\.consent"

:install
echo.
echo  [1/5] Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing via winget ^(may prompt for permission^)...
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] winget install failed.
        echo            Install Python manually from https://www.python.org/downloads/
        echo            ^(check "Add Python to PATH"^) and re-run this joiner.
        pause
        exit /b 1
    )
    rem refresh PATH for the rest of this session
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "PATH=%PATH%;%%b"
) else (
    echo        OK
)

echo  [2/5] Checking for Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing via winget...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] winget install failed.
        echo            Install Ollama manually from https://ollama.com/download
        echo            and re-run this joiner.
        pause
        exit /b 1
    )
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
) else (
    echo        OK
)

echo  [3/5] Ensuring Ollama daemon is running...
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    start "" /b ollama serve
    timeout /t 4 >nul
)
echo        OK

echo  [4/5] Ensuring model %MODEL% is pulled ^(~1 GB on first run^)...
ollama list 2>nul | findstr /b "%MODEL%" >nul
if errorlevel 1 (
    ollama pull %MODEL%
    if errorlevel 1 ( echo [!] model pull failed & pause & exit /b 1 )
) else (
    echo        OK
)

echo  [5/5] Downloading worker code...
curl -fsSL "%ORCH%/worker.py" -o "%WORKDIR%\worker.py"
if errorlevel 1 ( echo [!] download failed - is %ORCH% reachable? & pause & exit /b 1 )
curl -fsSL "%ORCH%/bundle.tar" -o "%WORKDIR%\bundle.tar"
if errorlevel 1 ( echo [!] bundle download failed & pause & exit /b 1 )
tar -xf "%WORKDIR%\bundle.tar" -C "%WORKDIR%"
if errorlevel 1 ( echo [!] extract failed & pause & exit /b 1 )

curl -fsSL "%ORCH%/stop.bat" -o "%WORKDIR%\stop.bat" >nul 2>nul
curl -fsSL "%ORCH%/chat.html" -o "%WORKDIR%\chat.html" >nul 2>nul
curl -fsSL "%ORCH%/nat-ai.bat" -o "%WORKDIR%\nat-ai.bat" >nul 2>nul

echo.
echo  ============================================================
echo   Setup complete!
echo.
echo   Opening:
echo     - NAT NETWORK NODE worker window  ^(your contribution^)
echo     - NAT AI chat in your browser     ^(your free local AI^)
echo.
echo   Close the worker window any time to pause contributing.
echo   Your local AI stays installed and usable in NAT AI chat.
echo   Re-open the chat any time by double-clicking:
echo     %WORKDIR%\nat-ai.bat
echo  ============================================================
echo.

start "NAT NETWORK NODE" cmd /k "title NAT NETWORK NODE && color 0B && cd /d %WORKDIR% && echo === NAT NETWORK NODE === && echo Server: %ORCH% && echo Close this window to stop contributing. && echo. && python -m worker --server %ORCH% --target ollama"

rem Give the worker ~6s to start its local NAT AI server on port 11500
timeout /t 6 >nul
start "" "http://127.0.0.1:11500/"

timeout /t 3 >nul
exit /b 0
"""

_STOP_BAT = r"""@echo off
title NAT NETWORK - Stop Node
echo Stopping NAT NETWORK worker windows...
for /f "tokens=2" %%a in ('tasklist /v /fi "WINDOWTITLE eq NAT NETWORK NODE*" /fo csv ^| findstr /v "PID"') do (
    taskkill /pid %%~a /t /f >nul 2>nul
)
echo Done. Your local model is unaffected; run the joiner again to rejoin.
pause
"""


def _render_joiner_bat(req) -> str:
    # CRLF line endings so cmd.exe is happy.
    return _JOINER_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_JOINER_COMMAND = r"""#!/bin/bash
# NAT NETWORK macOS one-click joiner. Double-click in Finder to run.
set -u
ORCH="{origin}"
WORKDIR="$HOME/.nat-network"
MODEL="qwen2.5-coder:1.5b"

clear
cat <<EOF

  ============================================================
                      N A T   N E T W O R K
                         Node Joiner (macOS)
  ============================================================

  Server: $ORCH
  Workdir: $WORKDIR

EOF

if [ -f "$WORKDIR/.consent" ]; then
  echo "  [Consent on file from $WORKDIR/.consent]"
else
  cat <<EOF
  Before you join, here is what this does:

    * Installs Homebrew (if missing), then Ollama via brew
    * Downloads a local AI model (~1 GB) you can use anytime
    * Opens a worker window that runs AI-safety evaluation jobs
      against your local Ollama, posting verdict labels back to NAT
    * Uses CPU/RAM/network while the worker window is open

  How to stop:
    * Close the "NAT NETWORK NODE" Terminal window any time
    * The local AI model stays installed and usable standalone

  Workload note: jobs include AdvBench safety probes. The local
  model only refuses or responds; nothing leaves your device
  beyond the verdict label.

EOF
  read -p "  Type Y to consent and join, or anything else to abort: " ans
  if [ "${{ans:-}}" != "Y" ] && [ "${{ans:-}}" != "y" ]; then
    echo "Aborted. Nothing installed."
    exit 1
  fi
  mkdir -p "$WORKDIR"
  printf 'consent_at=%s\norchestrator=%s\n' "$(date)" "$ORCH" > "$WORKDIR/.consent"
fi

echo
echo "  [1/5] Checking Homebrew..."
if ! command -v brew >/dev/null 2>&1; then
  echo "        Not found. Installing Homebrew (may ask for your sudo password)..."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [ $? -ne 0 ]; then
    echo "        [!] Homebrew install failed. Install manually from https://brew.sh then re-run."
    read -p "Press Enter to exit..." _
    exit 1
  fi
  # PATH for Apple Silicon
  if [ -x /opt/homebrew/bin/brew ]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
  # PATH for Intel Macs
  if [ -x /usr/local/bin/brew ]; then eval "$(/usr/local/bin/brew shellenv)"; fi
fi
echo "        OK"

echo "  [2/5] Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "        Not found. Installing via brew..."
  brew install ollama || {{ echo "        [!] brew install ollama failed"; read -p "Press Enter to exit..." _; exit 1; }}
fi
echo "        OK"

echo "  [3/5] Ensuring Ollama daemon is running..."
if ! curl -fsS -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/dev/null 2>&1 &)
  sleep 4
fi
echo "        OK"

echo "  [4/5] Ensuring model $MODEL is pulled (~1 GB on first run)..."
if ! ollama list 2>/dev/null | grep -q "^$MODEL"; then
  ollama pull "$MODEL" || {{ echo "[!] model pull failed"; read -p "Press Enter to exit..." _; exit 1; }}
else
  echo "        OK"
fi

echo "  [5/5] Downloading worker code..."
mkdir -p "$WORKDIR"
curl -fsSL "$ORCH/worker.py" -o "$WORKDIR/worker.py" || {{ echo "[!] worker.py download failed - is $ORCH reachable?"; read -p "Press Enter to exit..." _; exit 1; }}
curl -fsSL "$ORCH/bundle.tar" -o "$WORKDIR/bundle.tar"
tar -xf "$WORKDIR/bundle.tar" -C "$WORKDIR"
curl -fsSL "$ORCH/chat.html" -o "$WORKDIR/chat.html" 2>/dev/null
echo "        OK"

cat <<EOF

  ============================================================
   Setup complete!

   Opening:
     - NAT NETWORK NODE worker window  (your contribution)
     - NAT AI chat in your browser     (your free local AI)

   Close the worker window any time to pause contributing.
   Your local AI stays installed and usable.
  ============================================================

EOF

# Spawn the worker in a NEW Terminal window so it stays open after this script exits.
osascript <<APPLE
tell application "Terminal"
    activate
    do script "cd \"$WORKDIR\" && echo === NAT NETWORK NODE === && echo Server: $ORCH && echo Close this window to stop contributing. && echo && python3 -m worker --server $ORCH --target ollama"
end tell
APPLE

# Give the worker ~6s to start the local NAT AI HTTP server on 11500, then open chat.
sleep 6
open "http://127.0.0.1:11500/" 2>/dev/null

sleep 3
echo "You can close this window."
"""


def _render_joiner_command(req) -> str:
    # Use Python str.format but escape braces in shell ${...} by doubling them above.
    return _JOINER_COMMAND.format(origin=_origin(req))


_REACTIVATE_COMMAND = r"""#!/bin/bash
# NAT NETWORK macOS reconnect tool.
set -u
ORCH="{origin}"
WORKDIR="$HOME/.nat-network"

clear
cat <<EOF

  === NAT NETWORK reconnect (macOS) ===
  Server: $ORCH

EOF

if [ ! -f "$WORKDIR/worker.py" ]; then
  echo "  [!] No NAT install found at $WORKDIR"
  echo "      Run the full installer instead (joiner.command)."
  read -p "Press Enter to exit..." _
  exit 1
fi

echo "  [1/4] Stopping any old NAT NETWORK NODE windows..."
osascript -e 'tell application "Terminal" to close (every window whose name contains "NAT NETWORK NODE")' 2>/dev/null
echo "        OK"

echo "  [2/4] Ensuring Ollama is running..."
if ! curl -fsS -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1 || (nohup ollama serve >/dev/null 2>&1 &)
  sleep 4
fi
echo "        OK"

echo "  [3/4] Refreshing worker code from $ORCH..."
curl -fsSL "$ORCH/worker.py" -o "$WORKDIR/worker.py" 2>/dev/null
curl -fsSL "$ORCH/bundle.tar" -o "$WORKDIR/bundle.tar" 2>/dev/null
tar -xf "$WORKDIR/bundle.tar" -C "$WORKDIR" 2>/dev/null
curl -fsSL "$ORCH/chat.html" -o "$WORKDIR/chat.html" 2>/dev/null
echo "        OK"

echo "  [4/4] Launching worker + NAT AI chat..."
osascript <<APPLE
tell application "Terminal"
    activate
    do script "cd \"$WORKDIR\" && echo === NAT NETWORK NODE === && echo Server: $ORCH && echo && python3 -m worker --server $ORCH --target ollama"
end tell
APPLE
sleep 6
open "http://127.0.0.1:11500/" 2>/dev/null
echo
echo "  Reconnected. You can close this window."
sleep 4
"""


def _render_reactivate_command(req) -> str:
    return _REACTIVATE_COMMAND.format(origin=_origin(req))


_HANDOFF_COMMAND = r"""#!/bin/bash
# NAT NETWORK macOS project handoff: pull project tar, extract, copy kickoff prompt
# to clipboard, open Terminal in the project dir.
set -u
ORCH="{origin}"
DEST="$HOME/nat-network"

clear
cat <<EOF

  ============================================================
            NAT NETWORK - HANDOFF TO LOCAL CLAUDE
  ============================================================

  This will:
    1. Download the full NAT project from $ORCH
    2. Extract it to $DEST
    3. Copy the kickoff prompt to your clipboard
    4. Open a Terminal in the project folder

  When Claude is ready, paste with Cmd+V and press Enter.

EOF
read -p "  Type Y to continue, or anything else to abort: " ans
if [ "${{ans:-}}" != "Y" ] && [ "${{ans:-}}" != "y" ]; then exit 1; fi

mkdir -p "$DEST"
echo "  [1/3] Downloading project bundle..."
curl -fsSL "$ORCH/project.tar" -o "$DEST/project.tar" || {{ echo "[!] download failed - is $ORCH reachable from this Mac?"; read -p "Press Enter..." _; exit 1; }}
tar -xf "$DEST/project.tar" -C "$DEST"
rm -f "$DEST/project.tar"
echo "        OK"

echo "  [2/3] Copying kickoff prompt to clipboard..."
KICKOFF="Read HANDOFF.md and MEMORY/MEMORY.md and every file MEMORY.md references. After you've read them, give me a one-paragraph summary of where we left off so I know you're caught up. Then ask me what to work on next."
printf "%s" "$KICKOFF" | pbcopy
echo "        OK (clipboard now has the kickoff message)"

echo "  [3/3] Opening Terminal in $DEST..."
osascript <<APPLE
tell application "Terminal"
    activate
    do script "cd \"$DEST\" && echo Project ready. Run 'claude' here, then Cmd+V to paste the kickoff prompt."
end tell
APPLE
sleep 3
echo
echo "  Done. Switch to the new Terminal window."
sleep 3
"""


def _render_handoff_command(req) -> str:
    return _HANDOFF_COMMAND.format(origin=_origin(req))


def _render_stop_bat(_req) -> str:
    return _STOP_BAT.replace("\n", "\r\n")


_NAT_AI_BAT = r"""@echo off
title NAT AI - Launcher
echo Opening NAT AI chat in your default browser...
echo (served by the local NAT worker on port 11500)
echo If your browser shows an error, make sure the NAT NETWORK NODE window is running.
echo.
start "" "http://127.0.0.1:11500/"
timeout /t 3 >nul
exit /b 0
"""


def _render_nat_ai_bat(_req) -> str:
    return _NAT_AI_BAT.replace("\n", "\r\n")


_REACTIVATE_BAT = r"""@echo off
title NAT NETWORK - Reconnect
color 0B

set "ORCH={origin}"
set "WORKDIR=%USERPROFILE%\.nat-network"

cls
echo.
echo  === NAT NETWORK reconnect ===
echo  Server: %ORCH%
echo.

if not exist "%WORKDIR%\worker.py" (
    echo  [!] No NAT install found at %WORKDIR%
    echo      Run the full installer instead: download nat-network-joiner.bat
    pause
    exit /b 1
)

echo  [1/4] Stopping any old worker windows...
for /f "tokens=2" %%a in ('tasklist /v /fi "WINDOWTITLE eq NAT NETWORK NODE*" /fo csv ^| findstr /v "PID"') do (
    taskkill /pid %%~a /t /f >nul 2>nul
)
echo        OK

echo  [2/4] Ensuring Ollama is running...
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    start "" /b ollama serve
    timeout /t 4 >nul
)
echo        OK

echo  [3/4] Refreshing worker code from %ORCH%...
curl -fsSL "%ORCH%/worker.py" -o "%WORKDIR%\worker.py" >nul 2>nul
curl -fsSL "%ORCH%/bundle.tar" -o "%WORKDIR%\bundle.tar" >nul 2>nul
tar -xf "%WORKDIR%\bundle.tar" -C "%WORKDIR%" >nul 2>nul
curl -fsSL "%ORCH%/chat.html" -o "%WORKDIR%\chat.html" >nul 2>nul
echo        OK

echo  [4/4] Launching worker + NAT AI chat...
start "NAT NETWORK NODE" cmd /k "title NAT NETWORK NODE && color 0B && cd /d %WORKDIR% && echo === NAT NETWORK NODE === && echo Server: %ORCH% && echo. && python -m worker --server %ORCH% --target ollama"
timeout /t 6 >nul
start "" "http://127.0.0.1:11500/"

echo.
echo  Reconnected. You can close this window.
timeout /t 4 >nul
exit /b 0
"""


def _render_reactivate_bat(req) -> str:
    return _REACTIVATE_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_CODING_MODE_BAT = r"""@echo off
setlocal enabledelayedexpansion
title NAT NETWORK - Coding Mode Setup
color 0B

set "ORCH={origin}"
set "MODEL=llama3.1:8b"
set "CODE_DIR=%USERPROFILE%\nat-code"

cls
echo.
echo  ============================================================
echo                NAT NETWORK - CODING MODE
echo  ============================================================
echo.
echo   This installs the AI coding agent so NAT AI can actually
echo   read your files, propose edits, and run commands ^(with
echo   your approval each time^).
echo.
echo   Will install:
echo     * Aider ^(open-source AI pair-programmer^) via pip
echo     * %MODEL% local model ^(~4.7 GB, one-time download^)
echo.
echo   Will create a working folder at:
echo     %CODE_DIR%
echo.
echo   When done, double-click NAT CODE on your desktop or in
echo   %CODE_DIR%\nat-code.bat to start coding with the AI.
echo.

choice /c YN /n /m " Type Y to continue, or N to abort: "
if errorlevel 2 (
    echo Aborted.
    timeout /t 3 >nul
    exit /b 1
)

echo.
echo  [1/5] Checking Python 3.11 ^(Aider needs it for prebuilt wheels^)...
py -3.11 --version >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing Python 3.11 via winget...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] winget install failed. Install Python 3.11 from python.org and re-run.
        pause & exit /b 1
    )
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
    py -3.11 --version >nul 2>nul
    if errorlevel 1 (
        echo        [!] Python 3.11 installed but not on PATH yet. Close this window, open a new one, and run the bat again.
        pause & exit /b 1
    )
)
echo        OK

echo  [2/5] Checking for Git ^(required by Aider^)...
where git >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing via winget...
    winget install -e --id Git.Git --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] Git install failed. Install from https://git-scm.com/download/win then re-run this.
        pause & exit /b 1
    )
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
)
echo        OK

echo  [3/5] Installing Aider via pipx on Python 3.11 ^(isolated, no admin needed^)...
py -3.11 -m pip install --user --upgrade --quiet pipx
py -3.11 -m pipx ensurepath
rem Get the 3.11 user scripts dir for current session so `aider` resolves immediately.
for /f "delims=" %%a in ('py -3.11 -c "import sysconfig; print(sysconfig.get_path('scripts','nt_user'))" 2^>nul') do set "PATH=%%a;%PATH%"
py -3.11 -m pipx install aider-chat --force
if errorlevel 1 (
    echo.
    echo        [!] pipx install failed. Showing full output for debugging:
    echo.
    py -3.11 -m pipx install aider-chat --force --verbose
    echo.
    echo        Common fixes:
    echo          - "no module named pipx": close window, reopen, re-run this bat
    echo          - "ssl/certificate": Python install is broken; reinstall from python.org
    echo          - "build numpy": Python 3.11 wheel missing - retry with stable internet
    pause & exit /b 1
)
echo        OK

echo  [4/5] Checking model %MODEL%...
where ollama >nul 2>nul
if errorlevel 1 (
    echo        Ollama not found. Installing via winget...
    winget install -e --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 ( echo [!] winget install failed & pause & exit /b 1 )
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
)
tasklist /fi "imagename eq ollama.exe" 2>nul | find /i "ollama.exe" >nul
if errorlevel 1 (
    start "" /b ollama serve
    timeout /t 4 >nul
)
ollama list 2>nul | findstr /b "%MODEL%" >nul
if errorlevel 1 (
    echo        Pulling %MODEL% ^(~4.7 GB^)...
    ollama pull %MODEL%
    if errorlevel 1 ( echo [!] model pull failed & pause & exit /b 1 )
)
echo        OK

echo  [5/5] Creating working folder and desktop shortcut...
if not exist "%CODE_DIR%" mkdir "%CODE_DIR%"
curl -fsSL "%ORCH%/nat-code.bat" -o "%CODE_DIR%\nat-code.bat" >nul 2>nul
rem Simpler than a .lnk: just drop the .bat directly on the desktop.
copy /Y "%CODE_DIR%\nat-code.bat" "%USERPROFILE%\Desktop\NAT CODE.bat" >nul 2>nul
echo        OK

echo.
echo  ============================================================
echo   Coding mode ready. Launching now.
echo   Reopen later: double-click NAT CODE on your desktop, or
echo                 %CODE_DIR%\nat-code.bat
echo  ============================================================
echo.

start "" "%CODE_DIR%\nat-code.bat"
timeout /t 3 >nul
exit /b 0
"""


def _render_coding_mode_bat(req) -> str:
    return _CODING_MODE_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_NAT_CODE_BAT = r"""@echo off
title NAT CODE - AI Pair Programmer
color 0B
rem Aider was installed via pipx on Python 3.11 - use that scripts dir.
for /f "delims=" %%a in ('py -3.11 -c "import sysconfig; print(sysconfig.get_path('scripts','nt_user'))" 2^>nul') do set "PATH=%%a;%PATH%"
cd /d "%USERPROFILE%\nat-code"
echo.
echo  === NAT CODE ===
echo  Local AI pair programmer. Files in this folder are accessible.
echo  Drop files here, then type /add filename inside the chat.
echo.
echo  Tips:
echo    /help          - all commands
echo    /add file.py   - share a file with the AI
echo    /run cmd       - propose a shell command (AI asks before running)
echo    /undo          - revert last AI edit
echo    /exit          - quit
echo.
where aider >nul 2>nul
if errorlevel 1 (
    echo  [!] aider not on PATH. Try re-running the Coding Mode installer.
    pause
    exit /b 1
)
aider --model ollama/llama3.1:8b
pause
"""


def _render_nat_code_bat(_req) -> str:
    return _NAT_CODE_BAT.replace("\n", "\r\n")


_N8N_BAT = r"""@echo off
setlocal enabledelayedexpansion
title NAT NETWORK - Flow Setup (n8n)
color 0B

set "ORCH={origin}"
set "FLOW_DIR=%USERPROFILE%\nat-flow"

cls
echo.
echo  ============================================================
echo                NAT NETWORK - FLOW MODE (n8n)
echo  ============================================================
echo.
echo   This installs NAT FLOW, a visual workflow builder powered by
echo   n8n ^(open source^). Users build automations by drag-and-drop;
echo   workflows can call your local NAT AI through HTTP nodes.
echo.
echo   Will install:
echo     * Node.js LTS if missing ^(via winget^)
echo     * n8n via npx ^(cached, no global install needed^)
echo.
echo   Working folder ^(your flows live here^):
echo     %FLOW_DIR%
echo.
echo   Closing the NAT FLOW window stops the local server but your
echo   saved flows persist.
echo.

choice /c YN /n /m " Type Y to continue, or N to abort: "
if errorlevel 2 (
    echo Aborted.
    timeout /t 3 >nul
    exit /b 1
)

echo.
echo  [1/3] Checking Node.js...
where node >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing Node.js LTS via winget...
    winget install -e --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] winget install failed. Install Node.js from https://nodejs.org and re-run.
        pause & exit /b 1
    )
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
    where node >nul 2>nul
    if errorlevel 1 (
        echo        [!] Node installed but not on PATH. Close this window, reopen the installer.
        pause & exit /b 1
    )
)
echo        OK

echo  [2/3] Creating working folder and launcher...
if not exist "%FLOW_DIR%" mkdir "%FLOW_DIR%"
curl -fsSL "%ORCH%/nat-flow.bat" -o "%FLOW_DIR%\nat-flow.bat" >nul 2>nul
copy /Y "%FLOW_DIR%\nat-flow.bat" "%USERPROFILE%\Desktop\NAT FLOW.bat" >nul 2>nul
echo        OK

echo  [3/3] Pre-fetching n8n via npx ^(this can take a few minutes on first run^)...
set "N8N_USER_MANAGEMENT_DISABLED=true"
set "N8N_USER_FOLDER=%FLOW_DIR%"
rem Just verify npx can find it; the actual launch happens in nat-flow.bat
call npx --yes n8n --help >nul 2>nul

echo.
echo  ============================================================
echo   NAT FLOW ready. Launching now.
echo   Reopen later: double-click NAT FLOW on your desktop, or
echo                 %FLOW_DIR%\nat-flow.bat
echo.
echo   When n8n starts, your browser will open to
echo   http://localhost:5678 - that's the workflow editor.
echo.
echo   To call NAT AI from inside a workflow, add an HTTP Request
echo   node and POST to:  http://127.0.0.1:11500/api/chat
echo  ============================================================
echo.

start "" "%FLOW_DIR%\nat-flow.bat"
timeout /t 3 >nul
exit /b 0
"""


def _render_n8n_bat(req) -> str:
    return _N8N_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_NAT_FLOW_BAT = r"""@echo off
title NAT FLOW - Workflow Builder (n8n)
color 0B
set "N8N_USER_MANAGEMENT_DISABLED=true"
set "N8N_USER_FOLDER=%USERPROFILE%\nat-flow"
set "N8N_DIAGNOSTICS_ENABLED=false"
echo.
echo  === NAT FLOW ===
echo  Local workflow builder. Editor opens at http://localhost:5678
echo  Close this window to stop the local n8n server.
echo  Tip: HTTP Request node + http://127.0.0.1:11500/api/chat = call NAT AI from a workflow.
echo.
start "" "http://localhost:5678"
npx --yes n8n
pause
"""


def _render_nat_flow_bat(_req) -> str:
    return _NAT_FLOW_BAT.replace("\n", "\r\n")


_CLAUDE_INSTALL_BAT = r"""@echo off
setlocal enabledelayedexpansion
title NAT NETWORK - Claude Code Setup
color 0B

set "ORCH={origin}"
set "WORK_DIR=%USERPROFILE%\nat-claude"

cls
echo.
echo  ============================================================
echo               NAT NETWORK - CLAUDE CODE INSTALL
echo  ============================================================
echo.
echo   Installs Anthropic's official Claude Code CLI on this PC.
echo.
echo   You'll get NAT CLAUDE on your desktop. Double-click it to
echo   open a terminal with Claude ready to chat. On first run you
echo   sign in via your browser ^(Anthropic account, free or paid^).
echo.
echo   Note: this is the SAME Claude you may already be using
echo   elsewhere - the install just gives you a local CLI.
echo   API or subscription costs apply per Anthropic's pricing.
echo.

choice /c YN /n /m " Type Y to continue, or N to abort: "
if errorlevel 2 (
    echo Aborted.
    timeout /t 3 >nul
    exit /b 1
)

echo.
echo  [1/3] Checking Node.js ^(required by Claude Code^)...
where node >nul 2>nul
if errorlevel 1 (
    echo        Not found. Installing Node.js LTS via winget...
    winget install -e --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo        [!] winget install failed. Install Node.js from https://nodejs.org/ and re-run.
        pause & exit /b 1
    )
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "PATH=%%b;%PATH%"
    where node >nul 2>nul
    if errorlevel 1 (
        echo        [!] Node installed but not on PATH. Close this window and run the bat again.
        pause & exit /b 1
    )
)
echo        OK

echo  [2/3] Installing Claude Code via npm ^(global^)...
call npm install -g @anthropic-ai/claude-code
if errorlevel 1 (
    echo        [!] npm install failed. Run as administrator if you saw EACCES errors.
    pause & exit /b 1
)
echo        OK

echo  [3/3] Creating working folder and desktop launcher...
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"
curl -fsSL "%ORCH%/nat-claude.bat" -o "%WORK_DIR%\nat-claude.bat" >nul 2>nul
copy /Y "%WORK_DIR%\nat-claude.bat" "%USERPROFILE%\Desktop\NAT CLAUDE.bat" >nul 2>nul
echo        OK

echo.
echo  ============================================================
echo   Claude Code installed. Launching NAT CLAUDE now.
echo   Reopen later: double-click NAT CLAUDE on your desktop.
echo.
echo   First run will open a browser tab to sign in to Anthropic.
echo  ============================================================
echo.

start "" "%WORK_DIR%\nat-claude.bat"
timeout /t 3 >nul
exit /b 0
"""


def _render_claude_bat(req) -> str:
    return _CLAUDE_INSTALL_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_NAT_CLAUDE_LAUNCHER = r"""@echo off
title NAT CLAUDE
color 0B
cd /d "%USERPROFILE%\nat-claude"
echo.
echo  === NAT CLAUDE ===
echo  Anthropic Claude Code CLI. Working dir: %CD%
echo  First run opens browser to sign in.
echo  Type /help once you see the Claude prompt.
echo.
where claude >nul 2>nul
if errorlevel 1 (
    echo  [!] claude command not found. Re-run the installer or open a fresh terminal so npm's PATH takes effect.
    pause
    exit /b 1
)
claude
pause
"""


def _render_nat_claude_launcher(_req) -> str:
    return _NAT_CLAUDE_LAUNCHER.replace("\n", "\r\n")


_HANDOFF_BAT = r"""@echo off
setlocal enabledelayedexpansion
title NAT NETWORK - Project Handoff to Laptop Claude
color 0B

set "ORCH={origin}"
set "DEST=C:\nat-network"

cls
echo.
echo  ============================================================
echo            NAT NETWORK - HANDOFF TO LOCAL CLAUDE
echo  ============================================================
echo.
echo   This will:
echo     1. Download the full NAT project from %ORCH%
echo     2. Extract it to %DEST%
echo     3. Put the kickoff prompt on your clipboard
echo     4. Open NAT CLAUDE in the project folder
echo.
echo   When Claude is ready, just press Ctrl+V and Enter.
echo   The laptop Claude will read the handoff + memory files
echo   and pick up where the phone Claude left off.
echo.

choice /c YN /n /m " Type Y to continue, or N to abort: "
if errorlevel 2 ( echo Aborted. & timeout /t 2 >nul & exit /b 1 )

echo.
echo  [1/4] Downloading project bundle...
if not exist "%DEST%" mkdir "%DEST%"
curl -fsSL "%ORCH%/project.tar" -o "%DEST%\project.tar"
if errorlevel 1 ( echo [!] download failed - is %ORCH% reachable from this laptop? & pause & exit /b 1 )
echo        OK

echo  [2/4] Extracting...
tar -xf "%DEST%\project.tar" -C "%DEST%"
if errorlevel 1 ( echo [!] extract failed & pause & exit /b 1 )
del "%DEST%\project.tar" >nul 2>nul
echo        OK

echo  [3/4] Copying kickoff prompt to clipboard...
echo Read HANDOFF.md and MEMORY/MEMORY.md and every file MEMORY.md references. After you've read them, give me a one-paragraph summary of where we left off so I know you're caught up. Then ask me what to work on next. | clip
echo        OK ^(your clipboard now has the kickoff message^)

echo  [4/4] Launching NAT CLAUDE in %DEST%...
echo.
echo  ============================================================
echo   When the Claude prompt appears:
echo     1. Press Ctrl+V to paste the kickoff message
echo     2. Press Enter
echo   Claude will read everything and resume from where we
echo   left off on the phone.
echo  ============================================================
echo.
timeout /t 5 >nul

where claude >nul 2>nul
if errorlevel 1 (
    echo  [!] claude command not on PATH. Run the Claude Code installer first.
    pause
    exit /b 1
)

cd /d "%DEST%"
claude
pause
"""


def _render_handoff_bat(req) -> str:
    return _HANDOFF_BAT.format(origin=_origin(req)).replace("\n", "\r\n")


_CHAT_HTML = r"""<!doctype html>
<html><head>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>NAT AI</title>
<style>
  :root{--accent:#0bc;--bg:#0e1116;--fg:#e8eef5;--mute:#9aa4b2;--bub-u:#173039;--bub-a:#1a1f27}
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;flex-direction:column}
  .banner{background:#000;color:var(--accent);font-weight:700;letter-spacing:.2em;text-align:center;padding:.6em;font-size:.95em;border-bottom:1px solid #1a1f27}
  .topbar{display:flex;justify-content:space-between;align-items:center;padding:.6em 1em;background:#0b0f15;border-bottom:1px solid #1a1f27}
  .status{font-size:.85em;color:var(--mute)}
  .status.ok{color:#5cd693}
  .status.err{color:#ff6b6b}
  #chat{flex:1;overflow-y:auto;padding:1em;display:flex;flex-direction:column;gap:.8em;max-width:760px;width:100%;margin:0 auto}
  .msg-row{display:flex;gap:.6em;align-items:flex-start;max-width:85%}
  .msg-row.user{align-self:flex-end;flex-direction:row-reverse}
  .msg-row.assistant{align-self:flex-start}
  .msg-row.system{align-self:center;max-width:none}
  .avatar{width:36px;height:36px;border-radius:50%;flex-shrink:0;background:#1a1f27;border:1px solid #2b3340;overflow:hidden}
  .avatar img{width:100%;height:100%;display:block}
  .msg{padding:.7em 1em;border-radius:.6em;white-space:pre-wrap;word-wrap:break-word;line-height:1.5;flex:1;min-width:0}
  .msg.user{background:var(--bub-u);border:1px solid #244a59}
  .msg.assistant{background:var(--bub-a);border:1px solid #2b3340}
  .msg.system{background:transparent;color:var(--mute);font-size:.85em;font-style:italic;text-align:center}
  form{display:flex;gap:.5em;padding:1em;background:#0b0f15;border-top:1px solid #1a1f27;max-width:760px;width:100%;margin:0 auto}
  textarea{flex:1;resize:none;min-height:2.6em;max-height:8em;padding:.6em .8em;background:#1a1f27;color:var(--fg);border:1px solid #2b3340;border-radius:.4em;font:inherit}
  button{padding:0 1.2em;background:var(--accent);color:#000;border:0;border-radius:.4em;font-weight:700;cursor:pointer}
  button:disabled{opacity:.5;cursor:wait}
  small{color:var(--mute)}
  .hint{padding:.6em 1em;font-size:.85em;color:var(--mute);text-align:center}
  code{background:#1a1f27;padding:.1em .3em;border-radius:.3em;font-family:ui-monospace,Menlo,monospace}
  .contrib{background:#0f141b;border-bottom:1px solid #1a1f27;padding:.5em 1em;font-size:.85em}
  .contrib-summary{color:var(--mute);display:flex;justify-content:space-between;align-items:center;gap:.8em;flex-wrap:wrap}
  .counts{display:flex;gap:.6em;flex-wrap:wrap}
  .count{background:#1a1f27;padding:.15em .55em;border-radius:99em;border:1px solid #2b3340}
  .count.refuse{color:#5cd693}.count.partial{color:#ffd49a}.count.full{color:#ff6b6b}.count.ambig{color:#9aa4b2}.count.error{color:#ff6b6b}
  .contrib-feed{display:none;margin-top:.6em;max-height:8em;overflow-y:auto;font-family:ui-monospace,Menlo,monospace;font-size:.8em;color:var(--mute)}
  .contrib-feed.open{display:block}
  .contrib-row{padding:.1em 0;border-bottom:1px dotted #1a1f27}
  .contrib-toggle{cursor:pointer;color:var(--accent);font-weight:600;font-size:.8em;background:transparent;border:0;padding:0}
</style></head>
<body>
<div class=banner>
  <svg viewBox="0 0 28 28" width=22 height=22 style="vertical-align:-5px;margin-right:.5em;filter:drop-shadow(0 0 4px #0bc)" aria-hidden="true">
    <defs>
      <linearGradient id="ng" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0bc"/><stop offset="1" stop-color="#6f3"/></linearGradient>
    </defs>
    <circle cx="14" cy="14" r="3.2" fill="url(#ng)"/>
    <circle cx="5" cy="6" r="1.8" fill="#0bc"/>
    <circle cx="23" cy="6" r="1.8" fill="#0bc"/>
    <circle cx="5" cy="22" r="1.8" fill="#0bc"/>
    <circle cx="23" cy="22" r="1.8" fill="#0bc"/>
    <line x1="14" y1="14" x2="5" y2="6" stroke="#0bc" stroke-width="0.7" opacity="0.6"/>
    <line x1="14" y1="14" x2="23" y2="6" stroke="#0bc" stroke-width="0.7" opacity="0.6"/>
    <line x1="14" y1="14" x2="5" y2="22" stroke="#0bc" stroke-width="0.7" opacity="0.6"/>
    <line x1="14" y1="14" x2="23" y2="22" stroke="#0bc" stroke-width="0.7" opacity="0.6"/>
  </svg>NAT AI &middot; LOCAL MODEL
</div>
<div class=topbar>
  <div>Model: <code id=model>qwen2.5-coder:1.5b</code></div>
  <div>Mode:
    <select id=mode>
      <option value=plain>plain chat</option>
      <option value=node>node specialist</option>
      <option value=web3>web3 educator</option>
      <option value=code>explain code</option>
      <option value=debug>debug error</option>
      <option value=rewrite>quick rewrite</option>
      <option value=ideas>brainstorm</option>
      <option value=docs>docs prep</option>
    </select>
  </div>
  <div>Model:
    <select id=modelpick><option value=auto>auto</option></select>
  </div>
  <div class=status id=status>checking ollama...</div>
</div>
<div id=disclaimer style="display:none;background:#3a2a1a;color:#ffd49a;border-bottom:1px solid #5a4525;padding:.5em 1em;font-size:.85em;text-align:center">
  EDUCATION ONLY &middot; NAT AI is not a financial advisor. Verify everything in official docs. Never commit funds you cannot afford to lose.
</div>
<div class=contrib id=contribPanel>
  <div class=contrib-summary>
    <div>Your node contributions: <b id=ctotal>0</b></div>
    <div class=counts>
      <span class="count refuse">REFUSE: <span id=cR>0</span></span>
      <span class="count partial">PARTIAL: <span id=cP>0</span></span>
      <span class="count full">FULL: <span id=cF>0</span></span>
      <span class="count ambig">AMBIG: <span id=cA>0</span></span>
      <span class="count error">ERR: <span id=cE>0</span></span>
    </div>
    <button class=contrib-toggle id=ctoggle>show feed</button>
  </div>
  <div class=contrib-feed id=cfeed></div>
</div>
<div id=chat>
  <div class="msg system">NAT AI runs entirely on your machine via Ollama. Memory of this conversation is saved locally on this device. Nothing leaves your machine.</div>
</div>
<div style="padding:.4em 1em;font-size:.8em;color:var(--mute);display:flex;justify-content:space-between;align-items:center;gap:.5em;flex-wrap:wrap;border-bottom:1px solid #1a1f27;background:#0b0f15">
  <span>Memory: <span id=memstats>0 turns</span> <span id=memSummarized></span></span>
  <span>
    <button id=memClear class=contrib-toggle style="color:#ff9a9a">clear memory</button>
  </span>
</div>
<form id=form>
  <div style="display:flex;flex-direction:column;flex:1;gap:.4em">
    <details style="background:#1a1f27;border:1px solid #2b3340;border-radius:.4em;padding:.5em .7em">
      <summary style="cursor:pointer;color:var(--mute);font-size:.85em">paste file as context</summary>
      <textarea id=fileinput rows=4 placeholder="Paste a file's contents here. It'll be included as context for the next message." style="width:100%;margin-top:.5em;background:#0e1116;border:1px solid #2b3340;color:var(--fg);padding:.5em;border-radius:.3em;font-family:ui-monospace,Menlo,monospace;font-size:.85em"></textarea>
    </details>
    <textarea id=input placeholder="Talk to your local NAT AI..." autofocus></textarea>
  </div>
  <button id=send>Send</button>
</form>
<div class=hint id=hint></div>
<script>
// Empty base = relative URLs. Chat is served by the worker's local proxy,
// which forwards /api/* to Ollama. Same-origin, no CORS issues.
const OLLAMA_URL = "";
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const fileInput = document.getElementById('fileinput');
const modeSel = document.getElementById('mode');
const modelPick = document.getElementById('modelpick');
const disclaimer = document.getElementById('disclaimer');
const send = document.getElementById('send');
const statusEl = document.getElementById('status');
const modelEl = document.getElementById('model');
const hint = document.getElementById('hint');
const MEM_KEY = 'nat_ai_memory_v1';
const MEM_SUMMARY_KEY = 'nat_ai_memory_summary_v1';
const MEM_TURN_THRESHOLD = 30;   // start summarizing once we cross this many turns
const MEM_KEEP_RECENT = 12;       // keep this many most-recent turns verbatim
let history = JSON.parse(localStorage.getItem(MEM_KEY) || '[]');
let memorySummary = localStorage.getItem(MEM_SUMMARY_KEY) || '';
let MODEL = "qwen2.5-coder:1.5b";

// Per-device stable seed for the user's avatar. Generated once, persisted in localStorage.
const DEVICE_KEY = 'nat_ai_device_seed';
let deviceSeed = localStorage.getItem(DEVICE_KEY);
if (!deviceSeed) {
  deviceSeed = Math.random().toString(36).slice(2) + Date.now().toString(36);
  localStorage.setItem(DEVICE_KEY, deviceSeed);
}
const AVATAR_USER = `https://api.dicebear.com/8.x/pixel-art-neutral/svg?seed=${encodeURIComponent(deviceSeed)}&size=72&backgroundColor=1a1f27`;
const AVATAR_AI   = `https://api.dicebear.com/8.x/bottts-neutral/svg?seed=NAT-AI&size=72&backgroundColor=0c1a1f&primaryColor=00bbcc`;

function saveMemory() {
  localStorage.setItem(MEM_KEY, JSON.stringify(history));
  if (memorySummary) localStorage.setItem(MEM_SUMMARY_KEY, memorySummary);
  updateMemStats();
}
function updateMemStats() {
  const turns = history.length;
  document.getElementById('memstats').textContent = `${turns} turn${turns === 1 ? '' : 's'}`;
  document.getElementById('memSummarized').textContent = memorySummary ? `(plus summarized older history)` : '';
}
function restoreHistory() {
  if (!history.length && !memorySummary) return;
  if (memorySummary) {
    const d = document.createElement('div');
    d.className = 'msg system';
    d.textContent = '[Restored from memory — older conversation summarized]';
    chat.appendChild(d);
  }
  for (const m of history) {
    addMsg(m.role, m.content);
  }
}
async function summarizeOldHistory() {
  // Summarize everything except the last MEM_KEEP_RECENT turns. Replace them with one summary string.
  if (history.length <= MEM_TURN_THRESHOLD) return;
  const toSummarize = history.slice(0, history.length - MEM_KEEP_RECENT);
  const recent = history.slice(history.length - MEM_KEEP_RECENT);
  const prior = memorySummary ? `Prior summary: ${memorySummary}\n\n` : '';
  const sumPrompt = prior +
    "Below is older conversation between USER and ASSISTANT. Summarize the key facts, " +
    "decisions, code/files mentioned, and the user's preferences in <= 200 words. " +
    "Write in third person. Skip filler. Output only the summary, no preface.\n\n" +
    toSummarize.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        model: MODEL,
        messages: [{role: 'user', content: sumPrompt}],
        stream: false,
        keep_alive: "30m",
      }),
    });
    if (!r.ok) return;
    const j = await r.json();
    const summary = (j.message && j.message.content || '').trim();
    if (summary) {
      memorySummary = summary;
      history = recent;
      saveMemory();
      const d = document.createElement('div');
      d.className = 'msg system';
      d.textContent = '[Older conversation summarized to save memory]';
      chat.appendChild(d);
    }
  } catch {}
}

modeSel.addEventListener('change', () => {
  disclaimer.style.display = FINANCE_MODES.has(modeSel.value) ? '' : 'none';
});

const cfeed = document.getElementById('cfeed');
const ctoggle = document.getElementById('ctoggle');
ctoggle.addEventListener('click', () => {
  cfeed.classList.toggle('open');
  ctoggle.textContent = cfeed.classList.contains('open') ? 'hide feed' : 'show feed';
});
function fmtTs(ts){ const d=new Date(ts*1000); return d.toLocaleTimeString(); }
async function pollContribs(){
  try {
    const r = await fetch('/contributions');
    if (!r.ok) return;
    const d = await r.json();
    document.getElementById('ctotal').textContent = d.total || 0;
    document.getElementById('cR').textContent = (d.counts.REFUSE||0);
    document.getElementById('cP').textContent = (d.counts.COMPLY_PARTIAL||0);
    document.getElementById('cF').textContent = (d.counts.COMPLY_FULL||0);
    document.getElementById('cA').textContent = (d.counts.AMBIGUOUS||0);
    document.getElementById('cE').textContent = (d.counts.ERROR||0);
    cfeed.innerHTML = '';
    for (const c of [...(d.recent||[])].reverse()) {
      const row = document.createElement('div');
      row.className = 'contrib-row';
      row.textContent = `${fmtTs(c.ts)}  ${c.seed.padEnd(12)} ${c.chain.padEnd(40)} ${c.label}  c=${c.confidence}`;
      cfeed.appendChild(row);
    }
  } catch {}
}
pollContribs();
setInterval(pollContribs, 3000);

// Wire memory controls
document.getElementById('memClear').addEventListener('click', () => {
  if (confirm('Clear local NAT AI memory? This deletes the saved conversation on this device.')) {
    history = [];
    memorySummary = '';
    localStorage.removeItem(MEM_KEY);
    localStorage.removeItem(MEM_SUMMARY_KEY);
    chat.innerHTML = '<div class="msg system">Memory cleared.</div>';
    updateMemStats();
  }
});

// Restore prior session
restoreHistory();
updateMemStats();
modelPick.addEventListener('change', () => {
  if (modelPick.value !== 'auto') { MODEL = modelPick.value; modelEl.textContent = MODEL; }
});

const MODE_PROMPTS = {
  plain: "You are NAT AI, a local AI assistant running entirely on the user's machine. Be concise and direct.",
  node: "You are NAT AI in node-specialist mode. You help users set up, run, and troubleshoot compute nodes: validators, full nodes, miners, distributed-compute workers, RPC endpoints. You are strong on Linux and Windows admin, Docker, systemd, networking, port forwarding, dependency management, and reading error logs. When asked about specific blockchain implementations (Solana, Ethereum, Cosmos, etc.), explain general patterns confidently but defer to the project's official docs for version-specific commands and current parameters. Always note hardware and bandwidth implications when relevant. Be command-oriented when the user asks how-to questions.",
  web3: "You are NAT AI in web3-educator mode. The user is exploring an area they want to understand. Your job is to explain how protocols and mechanics WORK, never to recommend specific tokens, protocols, yields, or actions. For every topic raised, structure your response: 1) what it is, 2) how it works mechanically, 3) Pros, 4) Cons / risks, 5) where to verify. Always include a Risks section that lists realistic failure modes (smart-contract bugs, oracle failures, governance attacks, liquidity risk, regulatory risk, key management risk). Never predict prices or APYs. Never tell the user to do anything with money. If asked 'should I' or 'is X a good buy', decline and redirect to the conceptual underpinnings. End every response with: 'Education only — verify with official docs and only commit funds you can afford to lose.'",
  code: "You are NAT AI in code-explanation mode. The user will paste code or describe code. Explain what it does, line by line if helpful, and call out any bugs you spot. Be concise.",
  debug: "You are NAT AI in debug mode. The user will paste an error message or stack trace. Identify the most likely cause and suggest a specific fix. If more info is needed, ask one focused question.",
  rewrite: "You are NAT AI in rewrite mode. The user will paste text or a sentence. Rewrite it for clarity and brevity. Output the rewritten version only.",
  ideas: "You are NAT AI in brainstorm mode. The user will pose a question or problem. Generate 3-5 distinct ideas, each one sentence. No filler.",
  docs: "You are NAT AI in documentation-prep mode. The user gives you code, an API spec, a function, or a process. Output documentation in clean Markdown: a one-line summary, then sections for Purpose, Inputs, Outputs, Behavior, Errors / edge cases, and a minimal Example. Do not invent behavior not present in the input — if you don't see it, write 'not specified in source'. Be concise: no marketing language, no 'in conclusion' wrap-ups.",
};

const FINANCE_MODES = new Set(['web3']);

function addMsg(role, text){
  if (role === 'system') {
    // System messages stay simple (no avatar) — preserve old behavior
    const d = document.createElement('div');
    d.className = 'msg system';
    d.textContent = text;
    chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight;
    return d;
  }
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const av = document.createElement('div');
  av.className = 'avatar';
  const img = document.createElement('img');
  img.alt = role === 'user' ? 'you' : 'NAT AI';
  img.src = role === 'user' ? AVATAR_USER : AVATAR_AI;
  img.onerror = () => { img.style.display = 'none'; av.textContent = role === 'user' ? '👤' : '🤖'; av.style.display='flex'; av.style.alignItems='center'; av.style.justifyContent='center'; av.style.fontSize='20px'; };
  av.appendChild(img);
  const bubble = document.createElement('div');
  bubble.className = 'msg ' + role;
  bubble.textContent = text;
  row.appendChild(av);
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function checkOllama(){
  try {
    const r = await fetch(OLLAMA_URL + '/api/tags');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const models = (data.models||[]).map(m => m.name);
    if (!models.length) {
      statusEl.textContent = 'no models pulled';
      statusEl.className = 'status err';
      hint.innerHTML = 'Run <code>ollama pull qwen2.5-coder:1.5b</code> in a terminal.';
      return false;
    }
    if (!models.includes(MODEL)) MODEL = models[0];
    modelEl.textContent = MODEL;
    // populate the model dropdown
    modelPick.innerHTML = '';
    for (const m of models) {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === MODEL) opt.selected = true;
      modelPick.appendChild(opt);
    }
    statusEl.textContent = 'connected';
    statusEl.className = 'status ok';
    return true;
  } catch (e) {
    statusEl.textContent = 'cannot reach ollama';
    statusEl.className = 'status err';
    hint.innerHTML = 'Local NAT AI proxy not reachable. ' +
                     'Most likely the worker window was closed. ' +
                     'Open it (or run the reconnect tool) and reload this page.';
    return false;
  }
}

async function generate(userText){
  const mode = modeSel.value;
  const filePaste = fileInput.value.trim();
  // Build the user turn: if a file was pasted, prepend it as fenced context
  const userMsg = filePaste
    ? "Context file:\n```\n" + filePaste + "\n```\n\n" + userText
    : userText;
  // System message includes mode prompt + any prior summarized memory
  const sysContent = (MODE_PROMPTS[mode] || MODE_PROMPTS.plain) +
    (memorySummary ? `\n\n[Earlier conversation summary, treat as known context]: ${memorySummary}` : '');
  const sys = {role: 'system', content: sysContent};
  history.push({role:'user', content: userMsg});
  saveMemory();
  const placeholder = addMsg('assistant', '...');
  send.disabled = true;
  try {
    let r;
    // up to 2 retries if Ollama needs to reload the model (502 from proxy)
    for (let attempt = 0; attempt < 3; attempt++) {
      r = await fetch(OLLAMA_URL + '/api/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          model: MODEL,
          messages: [sys, ...history],
          stream: true,
          keep_alive: "30m",   // keep model resident; avoid reload-on-each-message
        }),
      });
      if (r.ok) break;
      if (r.status === 502 && attempt < 2) {
        placeholder.textContent = '(loading model, retrying...)';
        await new Promise(res => setTimeout(res, 3000));
        continue;
      }
      throw new Error('HTTP ' + r.status);
    }
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    placeholder.textContent = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, {stream:true});
      for (const line of chunk.split('\n')) {
        if (!line.trim()) continue;
        try {
          const j = JSON.parse(line);
          if (j.message && j.message.content) {
            text += j.message.content;
            placeholder.textContent = text;
            chat.scrollTop = chat.scrollHeight;
          }
        } catch {}
      }
    }
    history.push({role:'assistant', content: text});
    saveMemory();
    // After the turn, decide if memory needs trimming
    if (history.length > MEM_TURN_THRESHOLD) {
      summarizeOldHistory();
    }
  } catch (e) {
    placeholder.textContent = 'ERROR: ' + e.message;
    placeholder.style.color = '#ff6b6b';
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', e => {
  e.preventDefault();
  const t = input.value.trim();
  if (!t) return;
  input.value = '';
  addMsg('user', t);
  generate(t);
});

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});

checkOllama();
</script>
</body></html>
"""


def _render_chat_page() -> str:
    return _CHAT_HTML


def _read_worker_source() -> str:
    with open(os.path.join(os.path.dirname(__file__), "worker.py")) as f:
        return f.read()


def _build_bundle() -> bytes:
    """Build an in-memory tar of the `node/` package so a fresh worker host can run."""
    import io
    import tarfile
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    node_dir = os.path.join(repo_root, "node")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for dirpath, _, filenames in os.walk(node_dir):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if name.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, name)
                arc = os.path.relpath(full, repo_root)
                tar.add(full, arcname=arc)
    return buf.getvalue()


def _render_live_report(queue: "Queue") -> str:
    """Generate the customer-facing HTML safety report from completed probes."""
    import glob as _glob
    import sys as _sys
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    from node.jailbreak.report import load_probes, render_html
    paths = sorted(_glob.glob(os.path.join(queue.done_dir, "*.json")))
    probes = []
    for p in load_probes(paths):
        # done/ holds raw worker posts; keep only ones with a verdict
        if p.get("verdict"):
            probes.append(p)
    return render_html(probes)


def _build_full_project_bundle() -> bytes:
    """Tar of the entire agent-farm project (excluding caches, queue state, trajectories).

    Used by a fresh Claude Code session on a new machine to bootstrap with full project context.
    """
    import io
    import tarfile
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    skip_dirs = {"__pycache__", "state", "trajectories", ".git"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                if name.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, name)
                arc = os.path.relpath(full, repo_root)
                tar.add(full, arcname=arc)
    return buf.getvalue()


def _render_join_page(req) -> str:
    origin = _origin(req)
    return f"""<!doctype html>
<html><head>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>NAT NETWORK</title>
<style>
  :root{{--accent:#0bc;--bg:#0e1116;--fg:#e8eef5;--mute:#9aa4b2}}
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;line-height:1.55}}
  .banner{{position:sticky;top:0;background:#000;color:var(--accent);font-weight:700;letter-spacing:.2em;text-align:center;padding:.6em;font-size:.95em;border-bottom:1px solid #1a1f27;z-index:10}}
  .wrap{{max-width:720px;margin:1.5rem auto;padding:0 1rem}}
  code,pre{{background:#1a1f27;color:#cfe9f1;padding:.15em .35em;border-radius:.3em;font-family:ui-monospace,Menlo,monospace}}
  pre{{padding:1em;overflow:auto;white-space:pre-wrap;word-break:break-all;font-size:.92em}}
  .btn{{display:inline-block;padding:.7em 1.2em;background:var(--accent);color:#000;border-radius:.4em;text-decoration:none;cursor:pointer;border:0;font:inherit;font-weight:700}}
  .btn.alt{{background:#1a1f27;color:var(--fg);border:1px solid #2b3340}}
  .btn.big{{padding:1em 1.6em;font-size:1.05em}}
  .row{{display:flex;gap:.6em;flex-wrap:wrap;margin:.6em 0}}
  h1{{margin:.2em 0 .2em;font-size:1.6rem}}
  h2{{margin-top:1.6em}}
  small,.mute{{color:var(--mute)}}
  details{{margin:.8em 0;border:1px solid #1a1f27;border-radius:.4em;padding:.6em .8em;background:#0f141b}}
  details summary{{cursor:pointer;font-weight:600}}
  .pill{{display:inline-block;background:#1a1f27;color:var(--accent);padding:.15em .55em;border-radius:99em;font-size:.8em;border:1px solid #2b3340}}
  a{{color:var(--accent)}}
  .card{{background:#0f141b;border:1px solid #1a1f27;border-radius:.6em;padding:1.2em;margin:1em 0}}
</style></head><body>
<div class=banner>NODING ON NAT NETWORK</div>
<div class=wrap>
<h1>Join NAT NETWORK</h1>
<p class=mute>Run a node. Contribute compute. Get a free local AI agent.</p>

<div class=card>
<h2 style="margin-top:.2em">Windows — one tap</h2>
<ol>
  <li><a class="btn big" href="/joiner.bat" download>Download node installer (Windows)</a></li>
  <li>Open Downloads, double-click <code>nat-network-joiner.bat</code></li>
  <li>If Windows shows "Windows protected your PC" → <b>More info</b> → <b>Run anyway</b></li>
  <li>Read the consent screen, press <code>Y</code> to join</li>
  <li>The installer auto-installs Python and Ollama (if missing), pulls a free local AI model, then opens the <b>NAT NETWORK NODE</b> window</li>
</ol>
<p class=mute><small>Tested on Windows 10/11. Requires <code>winget</code> (built into recent Windows) — if winget is missing, the installer tells you where to get Python and Ollama manually.</small></p>
<p><a class="btn alt" href="/stop.bat" download>Download stop tool</a> <span class=mute><small>closes the worker; local model stays installed.</small></span></p>
</div>

<div class=card>
<h2 style="margin-top:.2em">macOS — one tap</h2>
<ol>
  <li><a class="btn big" href="/joiner-mac.zip" download>Download node installer (macOS)</a></li>
  <li>Open Downloads — the <code>.zip</code> auto-extracts to <code>nat-network-joiner.command</code></li>
  <li>Double-click the <code>.command</code> file. If macOS blocks it ("can't be opened because Apple cannot check it"), <b>right-click → Open → Open</b> once. After that double-click works.</li>
  <li>Read the consent prompt in Terminal, type <code>Y</code> and Enter</li>
  <li>Installer auto-installs Homebrew (if missing) and Ollama, pulls the model, opens the <b>NAT NETWORK NODE</b> Terminal window</li>
</ol>
<p class=mute><small>Tested on macOS 13+. Apple Silicon and Intel both supported. Reconnect tool: <a href="/reactivate-mac.zip">reactivate-mac.zip</a> &middot; Project handoff: <a href="/handoff-mac.zip">handoff-mac.zip</a>.</small></p>
</div>

<div class=card>
<h2 style="margin-top:.2em">Already installed? Reconnect</h2>
<p>If NAT is already installed on this PC and the worker / chat got closed, get them back in one click.</p>
<ol>
  <li><a class="btn big" href="/reactivate.bat" download>Download reconnect tool</a></li>
  <li>Double-click <code>nat-reactivate.bat</code> in your Downloads folder</li>
</ol>
<p class=mute><small>It refreshes the worker code, restarts Ollama with NAT AI enabled, and re-opens both windows. Skips consent/install if you've done them before.</small></p>
</div>

<h2>Other platforms</h2>
<details><summary>macOS / Linux / WSL — paste in terminal</summary>
<pre id=cmd>curl -fsSL {origin}/bootstrap.sh | bash</pre>
<div class=row>
  <button class=btn onclick="navigator.clipboard.writeText(document.getElementById('cmd').innerText);this.innerText='copied'">copy bash command</button>
  <a class="btn alt" href="/bootstrap.sh">view bootstrap.sh</a>
</div>
</details>

<details><summary>Windows PowerShell (if you prefer terminal)</summary>
<pre id=psh>iwr {origin}/bootstrap.ps1 -OutFile b.ps1; powershell -ExecutionPolicy Bypass -File .\\b.ps1</pre>
<div class=row>
  <button class=btn onclick="navigator.clipboard.writeText(document.getElementById('psh').innerText);this.innerText='copied'">copy PowerShell command</button>
  <a class="btn alt" href="/bootstrap.ps1">view bootstrap.ps1</a>
</div>
</details>

<details><summary>Any platform with Python</summary>
<pre id=py>python -c "import urllib.request; exec(urllib.request.urlopen('{origin}/bootstrap.py').read())"</pre>
<div class=row>
  <button class=btn onclick="navigator.clipboard.writeText(document.getElementById('py').innerText);this.innerText='copied'">copy Python command</button>
  <a class="btn alt" href="/bootstrap.py">view bootstrap.py</a>
</div>
</details>

<div class=card>
<h2 style="margin-top:.2em">What your node does</h2>
<ul>
  <li>Runs AI-safety evaluation jobs against a local AI model on your machine</li>
  <li>Sends only the <i>verdict label</i> back to NAT — your local prompts and responses stay on your device</li>
  <li>Uses CPU/RAM/network while the worker window is open</li>
  <li>Close the worker window any time to pause contributing</li>
  <li>The free local AI model remains installed and yours to use either way</li>
</ul>
</div>

<div class=card style="border-color:#0bc;background:#0c1a1f">
<h2 style="margin-top:.2em">Migrate project to this PC (with Claude memory)</h2>
<p>One-click: download the full NAT project tar from this orchestrator, extract to <code>C:\nat-network</code>, copy a kickoff prompt to your clipboard, and launch NAT CLAUDE in the project folder. Paste with Ctrl+V, hit Enter, and the laptop Claude picks up exactly where the phone Claude left off (reads HANDOFF.md and all memory files).</p>
<ol>
  <li>Make sure NAT CLAUDE is installed first (card below)</li>
  <li><a class="btn big" href="/handoff.bat" download>Download handoff tool</a></li>
  <li>Double-click <code>nat-handoff.bat</code>, press <code>Y</code></li>
  <li>When NAT CLAUDE opens, press <b>Ctrl+V</b> then <b>Enter</b></li>
</ol>
</div>

<div class=card>
<h2 style="margin-top:.2em">Install Claude Code (NAT CLAUDE)</h2>
<p>Anthropic's official AI coding agent, on your PC. Different from local NAT AI: uses Anthropic's Claude (more capable, requires Anthropic account, paid usage), runs in your terminal, can edit files and run commands across your codebase.</p>
<ol>
  <li><a class="btn big" href="/claude.bat" download>Download Claude Code installer</a></li>
  <li>Double-click <code>nat-claude-install.bat</code>, press <code>Y</code></li>
  <li>Installs Node.js (if missing), npm-installs Claude Code, drops a <b>NAT CLAUDE</b> shortcut on your desktop</li>
  <li>First launch opens a browser to sign in to your Anthropic account</li>
</ol>
<p class=mute><small>For maximum dev box: install Coding Mode (Aider, free local) AND NAT CLAUDE (Anthropic, paid but more capable). Use Aider for routine refactors; reach for Claude for harder design work.</small></p>
</div>

<div class=card>
<h2 style="margin-top:.2em">Add NAT Flow Mode (workflows)</h2>
<p>Build automations visually. NAT FLOW installs <b>n8n</b> (open-source workflow tool) and points it at your local AI. Use it for: scheduled tasks, watching folders, calling APIs, prepping documents, chaining AI prompts.</p>
<ol>
  <li><a class="btn big" href="/n8n.bat" download>Download Flow Mode installer</a></li>
  <li>Double-click <code>nat-n8n.bat</code> from Downloads, press <code>Y</code></li>
  <li>Installs Node.js (if missing), fetches n8n, drops a <b>NAT FLOW</b> shortcut on your desktop, opens the editor at <code>http://localhost:5678</code></li>
</ol>
<p class=mute><small>Inside n8n, call NAT AI via HTTP Request node → POST <code>http://127.0.0.1:11500/api/chat</code>. Workflows stay on your machine.</small></p>
</div>

<div class=card>
<h2 style="margin-top:.2em">Add NAT Coding Mode</h2>
<p>NAT AI chat is great for explaining + brainstorming. Add <b>NAT CODE</b> to let the AI read your files, propose diffs, and run commands (with your approval each time). One click, no terminal.</p>
<ol>
  <li><a class="btn big" href="/coding-mode.bat" download>Download Coding Mode installer</a></li>
  <li>Open Downloads, double-click <code>nat-coding-mode.bat</code></li>
  <li>Press <code>Y</code> when it asks to install</li>
  <li>It installs the coding agent + an 8B model (~5 GB, one-time), creates a working folder at <code>%USERPROFILE%\nat-code</code>, and puts a <b>NAT CODE</b> shortcut on your desktop</li>
</ol>
<p class=mute><small>Works alongside the NAT NETWORK NODE — same Ollama daemon, same private-by-default setup. Powered by Aider (open source).</small></p>
</div>

<p><a class="btn" href="/chat" target=_blank>NAT AI chat</a> &nbsp; <a class="btn alt" href="/status">live queue status</a> &nbsp; <span class=pill>server {origin}</span></p>
<p class=mute><small>NAT AI chat opens locally on whichever machine you opened this page from. It only works if Ollama is running and reachable at <code>127.0.0.1:11434</code> on that same machine.</small></p>
<p><small class=mute>The installer page does not auto-execute anything in the browser. Joining is opt-in and requires you to save and run a file.</small></p>
</div>
</body></html>
"""


def make_handler(queue: Queue):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode())

        def log_message(self, fmt, *args):  # silence default access log
            return

        def do_GET(self):
            if self.path == "/jobs/next":
                worker_id = self.headers.get("X-Worker-Id", "anon")
                job = queue.claim(worker_id)
                if job is None:
                    self.send_response(204)
                    self.end_headers()
                    return
                return self._send_json(200, job)
            if self.path == "/status":
                return self._send_json(200, queue.status())
            if self.path == "/report":
                return self._send_html(200, _render_live_report(queue))
            if self.path == "/" or self.path == "/join":
                return self._send_html(200, _render_join_page(self))
            if self.path == "/bootstrap.sh":
                return self._send_text(200, _render_bootstrap(self), content_type="text/x-shellscript")
            if self.path == "/bootstrap.ps1":
                return self._send_text(200, _render_bootstrap_ps1(self), content_type="text/plain")
            if self.path == "/bootstrap.py":
                return self._send_text(200, _render_bootstrap_py(self), content_type="text/x-python")
            if self.path == "/joiner.bat":
                return self._send_attachment(200, _render_joiner_bat(self), filename="nat-network-joiner.bat", content_type="application/x-bat")
            if self.path == "/joiner-mac.zip":
                return self._send_zip_attachment(200, "nat-network-joiner.command", _render_joiner_command(self),
                                                  zip_filename="nat-network-joiner.zip")
            if self.path == "/reactivate-mac.zip":
                return self._send_zip_attachment(200, "nat-reactivate.command", _render_reactivate_command(self),
                                                  zip_filename="nat-reactivate.zip")
            if self.path == "/handoff-mac.zip":
                return self._send_zip_attachment(200, "nat-handoff.command", _render_handoff_command(self),
                                                  zip_filename="nat-handoff.zip")
            if self.path == "/stop.bat":
                return self._send_attachment(200, _render_stop_bat(self), filename="nat-network-stop.bat", content_type="application/x-bat")
            if self.path == "/chat" or self.path == "/chat.html":
                return self._send_html(200, _render_chat_page())
            if self.path == "/nat-ai.bat":
                return self._send_attachment(200, _render_nat_ai_bat(self), filename="nat-ai.bat", content_type="application/x-bat")
            if self.path == "/reactivate.bat":
                return self._send_attachment(200, _render_reactivate_bat(self), filename="nat-reactivate.bat", content_type="application/x-bat")
            if self.path == "/coding-mode.bat":
                return self._send_attachment(200, _render_coding_mode_bat(self), filename="nat-coding-mode.bat", content_type="application/x-bat")
            if self.path == "/nat-code.bat":
                return self._send_attachment(200, _render_nat_code_bat(self), filename="nat-code.bat", content_type="application/x-bat")
            if self.path == "/n8n.bat":
                return self._send_attachment(200, _render_n8n_bat(self), filename="nat-n8n.bat", content_type="application/x-bat")
            if self.path == "/nat-flow.bat":
                return self._send_attachment(200, _render_nat_flow_bat(self), filename="nat-flow.bat", content_type="application/x-bat")
            if self.path == "/claude.bat":
                return self._send_attachment(200, _render_claude_bat(self), filename="nat-claude-install.bat", content_type="application/x-bat")
            if self.path == "/nat-claude.bat":
                return self._send_attachment(200, _render_nat_claude_launcher(self), filename="nat-claude.bat", content_type="application/x-bat")
            if self.path == "/handoff.bat":
                return self._send_attachment(200, _render_handoff_bat(self), filename="nat-handoff.bat", content_type="application/x-bat")
            if self.path == "/worker.py":
                return self._send_text(200, _read_worker_source(), content_type="text/x-python")
            if self.path == "/bundle.tar":
                return self._send_bytes(200, _build_bundle(), content_type="application/x-tar")
            if self.path == "/project.tar":
                return self._send_bytes(200, _build_full_project_bundle(), content_type="application/x-tar")
            self.send_response(404)
            self.end_headers()

        def _send_bytes(self, code: int, data: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_attachment(self, code: int, text: str, filename: str, content_type: str) -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_zip_attachment(self, code: int, inner_filename: str, inner_text: str, zip_filename: str) -> None:
            """Wrap a .command file in a ZIP with the executable bit set so macOS double-click works."""
            import io
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                info = zipfile.ZipInfo(inner_filename)
                # Unix file type (regular file = 0o100000) | mode 0o755 (rwx for owner, rx for others)
                info.external_attr = (0o100755 & 0xFFFF) << 16
                z.writestr(info, inner_text)
            body = buf.getvalue()
            self.send_response(code)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{zip_filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, code: int, html: str) -> None:
            body = html.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, code: int, text: str, content_type: str = "text/plain") -> None:
            body = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/jobs/submit":
                data = self._read_json()
                job_id = queue.submit(
                    behavior=data["behavior"],
                    chain=data.get("chain") or ["plain"],
                    seed_id=data.get("seed_id"),
                )
                return self._send_json(200, {"job_id": job_id})
            if self.path.startswith("/jobs/") and self.path.endswith("/result"):
                job_id = self.path.split("/")[2]
                probe = self._read_json()
                ok = queue.complete(job_id, probe)
                return self._send_json(200 if ok else 404, {"ok": ok})
            self.send_response(404)
            self.end_headers()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.path.join(os.path.dirname(__file__), "state"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lease-seconds", type=int, default=600)
    args = ap.parse_args()

    queue = Queue(state_dir=args.state_dir, lease_seconds=args.lease_seconds)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(queue))
    print(f"orchestrator listening on http://{args.host}:{args.port} state={args.state_dir}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
