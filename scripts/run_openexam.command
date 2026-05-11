#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "Starting OpenExam at http://127.0.0.1:8501"
echo "Close this window or press Ctrl+C to stop the service."

set +e
"$PYTHON" -m openexam ui --address 127.0.0.1 --port 8501
STATUS=$?
set -e

if [ "$STATUS" -eq 130 ]; then
  exit 130
fi

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "OpenExam failed to start."
  echo "Install dependencies first if needed:"
  echo '  python3 -m pip install -e ".[dev]"'
  echo
  read -r -p "Press Enter to close this window..."
  exit "$STATUS"
fi
