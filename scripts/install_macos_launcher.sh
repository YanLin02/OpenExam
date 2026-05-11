#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chmod +x "$SCRIPT_DIR/run_openexam.command"

echo "OpenExam macOS launcher is ready."
echo "You can double-click scripts/run_openexam.command to start the UI."
