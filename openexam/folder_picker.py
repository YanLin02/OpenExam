from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class FolderPickerResult:
    selected_path: str | None
    cancelled: bool
    error: str | None


def pick_folder_macos(prompt: str = "选择 OpenExam 资料目录") -> FolderPickerResult:
    escaped_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    script = f'POSIX path of (choose folder with prompt "{escaped_prompt}")'
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            shell=False,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return FolderPickerResult(None, False, "osascript is not available on this system.")
    except subprocess.TimeoutExpired:
        return FolderPickerResult(None, False, "Folder picker timed out.")
    except OSError as exc:
        return FolderPickerResult(None, False, str(exc))

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0 and stdout:
        return FolderPickerResult(stdout, False, None)

    lowered = stderr.casefold()
    if "user canceled" in lowered or "user cancelled" in lowered or "-128" in stderr:
        return FolderPickerResult(None, True, None)
    message = stderr or f"osascript exited with code {completed.returncode}."
    return FolderPickerResult(None, False, message)


def pick_folder(prompt: str = "选择 OpenExam 资料目录") -> FolderPickerResult:
    if platform.system() == "Darwin":
        return pick_folder_macos(prompt=prompt)
    return FolderPickerResult(
        None,
        False,
        "Folder picker is only implemented for macOS. Please enter the path manually.",
    )
