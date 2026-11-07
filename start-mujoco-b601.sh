#!/usr/bin/env bash
# First run installs dependencies and the official B601 assets into this folder.
set -eu

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3.10 or newer is required. Install Python 3 and its venv package, then retry.' >&2
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo 'Python 3.10 or newer is required. Check python3 --version.' >&2
  exit 1
fi
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo 'A Linux graphical desktop is required for the MuJoCo window. Start a desktop session and retry.' >&2
  exit 1
fi

if [ ! -e .venv ]; then
  echo 'Creating the Python environment...'
  if ! python3 -m venv .venv; then
    echo 'Could not create the environment. Install your distribution’s Python venv package, remove the incomplete .venv directory, and retry.' >&2
    exit 1
  fi
fi
PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
  echo 'The existing .venv is not a usable Linux environment. Remove it or move it aside, then retry.' >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo 'The existing .venv needs Python 3.10 or newer. Move it aside and retry.' >&2
  exit 1
fi
if ! "$PYTHON" -c 'import os, sys; sys.exit(not os.path.samefile(sys.prefix, ".venv"))'; then
  echo 'The existing .venv points to a different environment. Move it aside and retry to create a local one.' >&2
  exit 1
fi
if ! "$PYTHON" -m pip show arm-teleop-sdls >/dev/null 2>&1 || ! "$PYTHON" -m pip check >/dev/null 2>&1 || ! "$PYTHON" -c 'import numpy, mujoco' >/dev/null 2>&1; then
  echo 'Installing project dependencies (internet access required on the first run)...'
  if ! "$PYTHON" -m pip install -e .; then
    echo 'Dependency installation failed. Check internet access and Python/venv support, then retry.' >&2
    exit 1
  fi
fi
if ! "$PYTHON" -c 'import numpy, mujoco' >/dev/null 2>&1; then
  echo 'Dependencies are installed but cannot be loaded. Move .venv aside and retry.' >&2
  exit 1
fi

if [ ! -e robot_models/b601_rs ]; then
  echo 'Downloading and verifying the official B601 model for local use...'
  if ! "$PYTHON" scripts/install_b601.py; then
    echo 'Model installation failed. Check access to raw.githubusercontent.com, then retry.' >&2
    exit 1
  fi
elif [ ! -f robot_models/b601_rs/model.xml ] || [ ! -f robot_models/b601_rs/profile.json ] || [ ! -d robot_models/b601_rs/meshes ]; then
  echo 'The existing B601 model is incomplete. Move it aside before retrying; it will not be overwritten.' >&2
  exit 1
fi

# Pick an available localhost port. The server still reports an error if a
# different process takes it between this check and the server binding it.
PORT=$("$PYTHON" -c 'import socket
for port in range(8002, 8103):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("Ports 8002–8102 are in use. Close an existing server and retry.")') || exit 1
echo "Opening MuJoCo and http://127.0.0.1:$PORT/"
echo 'Keep this terminal open. Focus the browser page to control the arm.'
exec "$PYTHON" scripts/native_viewer.py --port "$PORT" --robot b601_rs --open-browser
