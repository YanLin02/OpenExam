from __future__ import annotations

import subprocess
from pathlib import Path


def file_uri(path: str | Path, page_number: int | None = None) -> str:
    uri = Path(path).expanduser().resolve().as_uri()
    if page_number is not None:
        return f"{uri}#page={page_number}"
    return uri


def open_local_file(path: str | Path) -> tuple[bool, str]:
    target = Path(path).expanduser()
    if not target.exists():
        return False, f"File does not exist: {target}"
    try:
        subprocess.Popen(
            ["open", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, f"Failed to open file: {exc}"
    return True, "Opening file."


def reveal_local_file(path: str | Path) -> tuple[bool, str]:
    target = Path(path).expanduser()
    if not target.exists():
        return False, f"File does not exist: {target}"
    try:
        subprocess.Popen(
            ["open", "-R", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, f"Failed to reveal file: {exc}"
    return True, "Revealing file in Finder."


def open_pdf_page_in_chrome(path: str | Path, page_number: int | None) -> tuple[bool, str]:
    target = Path(path).expanduser()
    if not target.exists():
        return False, f"File does not exist: {target}"
    if page_number is None:
        return open_local_file(target)
    uri = file_uri(target, page_number)
    try:
        subprocess.Popen(
            ["open", "-a", "Google Chrome", uri],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False, "未找到 Google Chrome，请使用页内预览或普通打开文件。"
    return True, "Opening PDF page in Google Chrome."
