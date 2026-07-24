#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BUILD_IF_MISSING=0
VERIFY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-if-missing)
      BUILD_IF_MISSING=1
      ;;
    --verify)
      VERIFY=1
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

APP_PATH="$PROJECT_ROOT/dist/OpenExam.app"
APP_SUPPORT_DIR="$HOME/Library/Application Support/OpenExam"
SERVER_JSON="$APP_SUPPORT_DIR/server.json"
LOG_PATH="$APP_SUPPORT_DIR/OpenExam.log"

if [ ! -d "$APP_PATH" ]; then
  if [ "$BUILD_IF_MISSING" -eq 1 ]; then
    "$SCRIPT_DIR/build_macos_app.sh"
  else
    echo "App not found: $APP_PATH" >&2
    echo "Build it first with: scripts/build_macos_app.sh" >&2
    exit 1
  fi
fi

/usr/bin/open -n "$APP_PATH"

if [ "$VERIFY" -eq 0 ]; then
  echo "Launched $APP_PATH"
  exit 0
fi

VERIFY_PYTHON="${PYTHON:-python3}"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  VERIFY_PYTHON="$PROJECT_ROOT/.venv/bin/python"
fi

deadline=$((SECONDS + 60))
while [ "$SECONDS" -lt "$deadline" ]; do
  URL="$("$VERIFY_PYTHON" -c 'import json, pathlib, sys; path = pathlib.Path(sys.argv[1]); print(json.loads(path.read_text(encoding="utf-8")).get("url", "")) if path.exists() else None' "$SERVER_JSON" 2>/dev/null || true)"
  if [ -n "$URL" ]; then
    if "$VERIFY_PYTHON" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=1).read(1)' "$URL" >/dev/null 2>&1; then
      echo "OpenExam is running at $URL"
      exit 0
    fi
  fi
  sleep 1
done

echo "OpenExam did not become reachable within 60 seconds." >&2
if [ -f "$LOG_PATH" ]; then
  echo "Recent log output:" >&2
  tail -n 80 "$LOG_PATH" >&2
fi
exit 1
