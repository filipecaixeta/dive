#!/usr/bin/env bash
# Starts Dive Color at http://localhost:8765
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  PY=$(command -v python3.12 || command -v python3.13 || command -v python3)
  "$PY" -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
# GoPro's official converter for RAW photos (.GPR); built once if cmake is available
if [ ! -x tools/bin/gpr_tools ] && command -v cmake >/dev/null && command -v git >/dev/null; then
  echo "Building gpr_tools (GoPro RAW)..."
  mkdir -p tools/bin
  [ -d tools/gpr-src ] || git clone --depth 1 -q https://github.com/gopro/gpr.git tools/gpr-src
  cmake -S tools/gpr-src -B tools/gpr-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null
  cmake --build tools/gpr-build -j 8 >/dev/null
  cp tools/gpr-build/source/app/gpr_tools/gpr_tools tools/bin/ && rm -rf tools/gpr-build
fi
exec .venv/bin/uvicorn server:app --app-dir backend --port "${PORT:-8765}"
