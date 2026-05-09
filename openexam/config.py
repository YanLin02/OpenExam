from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


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
    semantic_candidate_limit: int = 100
    fuzzy_scan_limit: int = 50000
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_batch_size: int = 16

    @property
    def db_path(self) -> Path:
        return self.index_dir / self.db_name

    @property
    def embedding_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedding_model)

    @property
    def embeddings_npy_path(self) -> Path:
        return self.index_dir / f"embeddings_{self.embedding_slug}.npy"

    @property
    def embeddings_json_path(self) -> Path:
        return self.index_dir / f"embeddings_{self.embedding_slug}.json"


DEFAULT_CONFIG = AppConfig()
