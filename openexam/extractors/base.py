from __future__ import annotations

from pathlib import Path
from typing import Protocol

from openexam.models import ExtractedSection


class Extractor(Protocol):
    def extract(self, path: Path) -> list[ExtractedSection]:
        """Extract text sections while preserving source location metadata."""


class ExtractionError(RuntimeError):
    pass
