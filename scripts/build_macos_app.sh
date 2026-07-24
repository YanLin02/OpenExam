#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${OPENEXAM_BUILD_VENV:-.venv}"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$PROJECT_ROOT/.pyinstaller}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install -e ".[dev,macos-app]"
"$VENV_PYTHON" -m pytest
"$VENV_PYTHON" -m PyInstaller --clean --noconfirm OpenExam.spec

APP_PATH="$PROJECT_ROOT/dist/OpenExam.app"
APP_EXECUTABLE="$APP_PATH/Contents/MacOS/OpenExam"
INFO_PLIST="$APP_PATH/Contents/Info.plist"

if [ ! -x "$APP_EXECUTABLE" ]; then
  echo "Expected app executable was not created: $APP_EXECUTABLE" >&2
  exit 1
fi

if [ ! -f "$INFO_PLIST" ]; then
  echo "Expected Info.plist was not created: $INFO_PLIST" >&2
  exit 1
fi

echo "Built $APP_PATH"
