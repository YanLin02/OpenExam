from __future__ import annotations

from pathlib import Path

from openexam.ask import ask_question
from openexam.config import AppConfig
from openexam.db import connect
from openexam.models import SearchResult
from openexam.priority_sources import (
    find_indexed_answer_bank_sources,
    is_heading_only_answer_bank_chunk,
    is_exam_answer_bank_path,
    merge_priority_results,
    priority_source_label,
    split_answer_bank_markdown_sections,
)
from openexam.solve import solve_question


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
        text="text",
        snippet="text",
        score=100.0 - chunk_db_id,
        mode="hybrid",
    )


def test_exam_answer_bank_path_detection() -> None:
    assert is_exam_answer_bank_path("/data/深度学习简答题_开卷检索版.md")
    assert is_exam_answer_bank_path("/data/近五年真题.md")
    assert is_exam_answer_bank_path("/data/《深度学习》附录名词术语详解.pdf")
    assert not is_exam_answer_bank_path("/data/Chapter+2-CNN.pdf")
    assert priority_source_label("/data/深度学习简答题_开卷检索版.md") == "short_answer_bank"
    assert priority_source_label("/data/近五年真题.md") == "past_exam_bank"
    assert priority_source_label("/data/《深度学习》附录名词术语详解.pdf") == "terminology_bank"


def test_merge_priority_results_orders_answer_bank_first() -> None:
    regular_1 = make_result(1, "/data/Chapter+2-CNN.pdf")
    priority_1 = make_result(2, "/data/近五年真题.md")
    regular_2 = make_result(3, "/data/Chapter+8.pdf")
    priority_2 = make_result(4, "/data/深度学习简答题_开卷检索版.md")

    merged = merge_priority_results([regular_1, priority_1, regular_2, priority_2], top_k=4, priority_enabled=True)

    assert merged == [priority_1, priority_2, regular_1, regular_2]


def test_merge_priority_results_preserves_original_order_and_dedupes() -> None:
    priority_1 = make_result(1, "/data/近五年真题.md")
    priority_2 = make_result(2, "/data/深度学习简答题_开卷检索版.md")
    duplicate = priority_1.model_copy()

    merged = merge_priority_results([priority_1, priority_2, duplicate], top_k=5, priority_enabled=True)

    assert merged == [priority_1, priority_2]
    assert merge_priority_results([priority_2, priority_1], top_k=1, priority_enabled=False) == [priority_2]


def test_split_answer_bank_markdown_sections_merges_heading_and_answer() -> None:
    text = """## 第二章 生成模型

### 5．请简述 GAN 的训练过程。

GAN 由生成器和判别器组成。

训练过程：
1. 固定 G，训练 D；
2. 固定 D，训练 G。

### 6．请简述 Dropout。

Dropout 是一种正则化方法。
"""

    sections = split_answer_bank_markdown_sections(text)

    assert len(sections) == 2
    assert "### 5．请简述 GAN 的训练过程。" in sections[0]
    assert "生成器" in sections[0]
    assert "判别器" in sections[0]
    assert "固定 G" in sections[0]
    assert "### 6．请简述 Dropout。" in sections[1]
    assert "Dropout 是一种正则化方法" in sections[1]


def test_split_answer_bank_markdown_sections_ignores_chapter_heading_only() -> None:
    sections = split_answer_bank_markdown_sections(
        """## 第二章 卷积神经网络

### 1. 深度学习（Deep Learning）

深度学习是表示学习的一类方法。
"""
    )

    assert sections == ["### 1. 深度学习（Deep Learning）\n\n深度学习是表示学习的一类方法。"]


def test_heading_only_answer_bank_chunk_detection() -> None:
    assert is_heading_only_answer_bank_chunk("### 5．请简述 GAN 的训练过程。")
    assert is_heading_only_answer_bank_chunk("### 5．请简述 GAN 的训练过程。\n\n答案：")
    assert not is_heading_only_answer_bank_chunk("### 6．请简述 Dropout。\n\nDropout 是一种正则化方法。")


def test_merge_priority_results_skips_heading_only_priority_chunk() -> None:
    heading = make_result(1, "/data/近五年真题.md")
    heading.text = "### 5．请简述 GAN 的训练过程。"
    answer = make_result(2, "/data/深度学习简答题_开卷检索版.md")
    answer.text = "### 5．请简述 GAN 的训练过程。\n\nGAN 由生成器和判别器组成。"
    regular = make_result(3, "/data/Chapter+2-CNN.pdf")

    merged = merge_priority_results([regular, heading, answer], top_k=3, priority_enabled=True)

    assert merged == [answer, regular]
    assert heading not in merged


def test_find_indexed_answer_bank_sources(tmp_path: Path) -> None:
    config = AppConfig(index_dir=tmp_path / ".openexam")
    conn = connect(config.db_path)
    try:
        conn.execute(
            "INSERT INTO documents(path, filename, ext, status, source_type) VALUES (?, ?, ?, ?, ?)",
            ("/data/近五年真题.md", "近五年真题.md", ".md", "indexed", "other"),
        )
        conn.execute(
            "INSERT INTO documents(path, filename, ext, status, source_type) VALUES (?, ?, ?, ?, ?)",
            ("/data/Chapter+2-CNN.pdf", "Chapter+2-CNN.pdf", ".pdf", "indexed", "lecture"),
        )
        conn.commit()
    finally:
        conn.close()

    assert find_indexed_answer_bank_sources(config) == ["近五年真题.md"]


def test_ask_question_passes_priority_to_search(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_search_index(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("openexam.ask.search_index", fake_search_index)
    response = ask_question(
        "什么是 Dropout？",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        evidence_policy="strict",
        priority_answer_bank=True,
    )

    assert seen["priority_answer_bank"] is True
    assert response.priority_answer_bank is True
    assert response.llm_called is False


def test_solve_concept_defaults_to_priority_answer_bank(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_ask_question(question, **kwargs):
        seen.update(kwargs)
        from openexam.ask import AskResponse

        return AskResponse(
            question=question,
            answer="fake",
            results=[],
            search_mode=kwargs["mode"],
            scope=kwargs["scope"],
            prefer=kwargs["prefer"],
            per_file_cap=kwargs["per_file_cap"],
            top_k=kwargs["top_k"] or 6,
            llm_model=kwargs["llm_model"] or "qwen3:8b",
            llm_called=True,
            evidence_status="sufficient",
            evidence_policy=kwargs["evidence_policy"],
            missing_phrases=[],
            timing={"total_time_ms": 1.0},
            detail=kwargs["detail"],
            priority_answer_bank=kwargs["priority_answer_bank"],
        )

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)
    response = solve_question("什么是 Dropout？", config=AppConfig(index_dir=tmp_path / ".openexam"))

    assert seen["priority_answer_bank"] is True
    assert response.ask_response.priority_answer_bank is True


def test_solve_calculation_ignores_priority_answer_bank(monkeypatch, tmp_path: Path) -> None:
    def fail_if_ask_called(*args, **kwargs):
        raise AssertionError("calculator should not call ask_question")

    monkeypatch.setattr("openexam.solve.ask_question", fail_if_ask_called)
    response = solve_question(
        "给定输入 32x32，卷积核 5x5，stride=1，padding=0，输出尺寸是多少？",
        mode="calculation",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        priority_answer_bank=True,
    )

    assert response.ask_response.llm_called is False
    assert response.ask_response.priority_answer_bank is False
    assert response.calculation_answer is not None
