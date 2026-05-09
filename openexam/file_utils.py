from __future__ import annotations

import subprocess
from pathlib import Path


def file_uri(path: str | Path, page_number: int | None = None) -> str:
    uri = Path(path).expanduser().resolve().as_uri()
    if page_number is not None:
        return f"{uri}#page={page_number}"
    return uri


def open_local_file(path: str | Path) -> tuple[bool, str]:
    try:
        subprocess.run(["open", str(Path(path).expanduser())], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"Failed to open file: {exc}"
    return True, "Opened file."
