from __future__ import annotations

from pathlib import Path
from typing import Literal


SourceType = Literal["lecture", "textbook_ocr", "other"]
SearchScope = Literal["all", "lecture", "textbook_ocr", "other"]
SourcePreference = Literal["none", "lecture", "textbook_ocr"]


def classify_source_type(path_or_name: str | Path) -> SourceType:
    name = Path(path_or_name).name if isinstance(path_or_name, Path) else Path(str(path_or_name)).name
    lowered = name.lower()
    normalized = lowered.replace("+", " ").replace("_", " ").replace("-", " ")
    if "ocr" in normalized or "layered" in normalized:
        return "textbook_ocr"
    lecture_markers = ("chapter", "course overview", "附录", "课件")
    if any(marker in normalized for marker in lecture_markers):
        return "lecture"
    return "other"


def scope_matches(source_type: str, scope: SearchScope) -> bool:
    return scope == "all" or source_type == scope
