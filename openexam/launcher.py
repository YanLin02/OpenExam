from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def streamlit_command(address: str = "127.0.0.1", port: int = 8501, headless: bool = False) -> list[str]:
    app_path = Path(__file__).resolve().with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        address,
        "--server.port",
        str(port),
    ]
    if headless:
        command.extend(["--server.headless", "true"])
    return command


def launch_streamlit(
    address: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = False,
) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(
            "Streamlit is not installed. Install OpenExam dependencies first:\n"
            '  python3 -m pip install -e ".[dev]"',
            file=sys.stderr,
        )
        return 2

    try:
        return subprocess.call(streamlit_command(address=address, port=port, headless=headless))
    except KeyboardInterrupt:
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openexam-ui", description="Start the OpenExam Streamlit UI.")
    parser.add_argument("--address", default="127.0.0.1", help="Address for the Streamlit server. Default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit server. Default: 8501.")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return launch_streamlit(address=args.address, port=args.port, headless=args.headless)
