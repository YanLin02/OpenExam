from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}


@dataclass(frozen=True)
class AppConfig:
    index_dir: Path = Path(".openexam")
    db_name: str = "index.sqlite3"
    chunk_size: int = 1000
    chunk_overlap: int = 120
    default_top_k: int = 10
    fts_candidate_limit: int = 100
    fuzzy_candidate_limit: int = 100
    fuzzy_scan_limit: int = 50000

    @property
    def db_path(self) -> Path:
        return self.index_dir / self.db_name


DEFAULT_CONFIG = AppConfig()
