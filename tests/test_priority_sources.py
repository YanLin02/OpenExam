from __future__ import annotations

from pathlib import Path

from openexam.config import AppConfig
from openexam.db import connect
from openexam.models import SearchResult
from openexam.priority_sources import (
    DEFAULT_PRIORITY_PATTERNS,
    default_priority_source_config,
    find_indexed_answer_bank_sources,
    is_heading_only_answer_bank_chunk,
    is_exam_answer_bank_path,
    load_priority_source_config,
    merge_priority_results,
    priority_source_label,
    split_answer_bank_markdown_sections,
)


def make_result(chunk_db_id: int, path: str) -> SearchResult:
    return SearchResult(
        chunk_db_id=chunk_db_id,
        document_id=chunk_db_id,
        file_name=Path(path).name,
        source_path=path,
        source_type="other",
        location_type="paragraph",
        location_label="paragraph 1",
        paragraph_index=1,
        chunk_id=f"chunk-{chunk_db_id}",
        text="answer text",
        snippet="answer text",
        score=100.0 - chunk_db_id,
        mode="hybrid",
    )


def test_default_detection_uses_directories_only() -> None:
    priority_config = default_priority_source_config()

    assert DEFAULT_PRIORITY_PATTERNS == ()
    assert priority_config.priority_patterns == ()
    assert is_exam_answer_bank_path("/data/answer_bank/concepts.md", priority_config)
    assert is_exam_answer_bank_path("/data/exam_answer_bank/concepts.md", priority_config)
    assert is_exam_answer_bank_path("/data/priority_sources/concepts.md", priority_config)
    assert is_exam_answer_bank_path("/data/易考/concepts.md", priority_config)
    assert is_exam_answer_bank_path("/data/重点/concepts.md", priority_config)
    assert is_exam_answer_bank_path("/data/答案库/concepts.md", priority_config)
    assert not is_exam_answer_bank_path("/data/concepts_answer_bank_notes.md", priority_config)
    assert not is_exam_answer_bank_path("/data/regular_notes.md", priority_config)
    assert priority_source_label("/data/answer_bank/concepts.md", priority_config) == "directory_answer_bank"
    assert priority_source_label("/data/易考/concepts.md", priority_config) == "exam_focus_bank"
    assert priority_source_label("/data/regular_notes.md", priority_config) is None


def test_user_config_merges_patterns_and_labels(tmp_path: Path) -> None:
    config = AppConfig(index_dir=tmp_path / ".openexam")
    config.index_dir.mkdir()
    (config.index_dir / "priority_sources.json").write_text(
        """
        {
          "answer_bank_dirs": ["my_bank"],
          "priority_patterns": ["curated_notes"],
          "labels": {
            "my_bank": "personal_bank",
            "curated_notes": "curated_bank"
          }
        }
        """,
        encoding="utf-8",
    )

    priority_config = load_priority_source_config(config)

    assert is_exam_answer_bank_path("/data/my_bank/a.md", priority_config)
    assert is_exam_answer_bank_path("/data/curated_notes.md", priority_config)
    assert priority_source_label("/data/my_bank/a.md", priority_config) == "personal_bank"
    assert priority_source_label("/data/curated_notes.md", priority_config) == "curated_bank"


def test_broken_user_config_falls_back_to_defaults(tmp_path: Path) -> None:
    config = AppConfig(index_dir=tmp_path / ".openexam")
    config.index_dir.mkdir()
    (config.index_dir / "priority_sources.json").write_text("{broken", encoding="utf-8")

    priority_config = load_priority_source_config(config)

    assert "answer_bank" in priority_config.answer_bank_dirs
    assert priority_config.priority_patterns == ()
    assert priority_config.warnings


def test_split_answer_bank_markdown_sections_merges_heading_and_body() -> None:
    text = """## Chapter

### What is dropout?

Dropout randomly disables activations during training.

### What is attention?

Attention weights relationships between tokens.
"""

    sections = split_answer_bank_markdown_sections(text)

    assert len(sections) == 2
    assert sections[0].startswith("### What is dropout?")
    assert "randomly disables" in sections[0]
    assert sections[1].startswith("### What is attention?")
    assert "relationships" in sections[1]


def test_heading_only_answer_bank_chunk_detection() -> None:
    assert is_heading_only_answer_bank_chunk("### What is dropout?")
    assert is_heading_only_answer_bank_chunk("### What is dropout?\n\nAnswer:")
    assert not is_heading_only_answer_bank_chunk("### What is dropout?\n\nDropout is a regularizer.")


def test_merge_priority_results_orders_answer_bank_first_and_dedupes() -> None:
    regular = make_result(1, "/data/notes.md")
    priority = make_result(2, "/data/answer_bank/concepts.md")
    duplicate = priority.model_copy()

    merged = merge_priority_results([regular, priority, duplicate], top_k=3, priority_enabled=True)

    assert merged == [priority, regular]
    assert merge_priority_results([regular, priority], top_k=2, priority_enabled=False) == [regular, priority]


def test_merge_priority_results_skips_heading_only_priority_chunks() -> None:
    heading = make_result(1, "/data/answer_bank/concepts.md")
    heading.text = "### What is dropout?"
    answer = make_result(2, "/data/answer_bank/concepts.md")
    answer.text = "### What is dropout?\n\nDropout is a regularizer."
    regular = make_result(3, "/data/notes.md")

    merged = merge_priority_results([regular, heading, answer], top_k=3, priority_enabled=True)

    assert merged == [answer, regular]


def test_find_indexed_answer_bank_sources(tmp_path: Path) -> None:
    config = AppConfig(index_dir=tmp_path / ".openexam")
    conn = connect(config.db_path)
    try:
        conn.execute(
            "INSERT INTO documents(path, filename, ext, status, source_type) VALUES (?, ?, ?, ?, ?)",
            ("/data/answer_bank/concepts.md", "concepts.md", ".md", "indexed", "other"),
        )
        conn.execute(
            "INSERT INTO documents(path, filename, ext, status, source_type) VALUES (?, ?, ?, ?, ?)",
            ("/data/notes.md", "notes.md", ".md", "indexed", "other"),
        )
        conn.commit()
    finally:
        conn.close()

    assert find_indexed_answer_bank_sources(config) == ["concepts.md [directory_answer_bank]"]
