#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ ! -x .venv/bin/python ]; then
  echo 'Run bash start-mujoco-b601.sh first to set up the environment.' >&2
  exit 1
fi
exec .venv/bin/python scripts/replay_episode.py --latest
