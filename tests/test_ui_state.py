from __future__ import annotations

from openexam.ui_state import (
    build_ask_signature,
    build_search_signature,
    compact_index_status,
    compact_ollama_status,
    compact_source_status,
    preview_state_key,
    preview_toggle_label,
)


def test_ask_signature_is_stable_and_sorted() -> None:
    first = build_ask_signature(
        query="Transformer 中注意力机制的作用",
        mode="hybrid",
        scope="all",
        prefer="lecture",
        per_file_cap=2,
        top_k=6,
        llm_model="qwen3:8b",
        evidence_policy="warn",
        detail="standard",
    )
    second = build_ask_signature(
        detail="standard",
        evidence_policy="warn",
        llm_model="qwen3:8b",
        top_k=6,
        per_file_cap=2,
        prefer="lecture",
        scope="all",
        mode="hybrid",
        query="Transformer 中注意力机制的作用",
    )

    assert first == second
    assert '"query": "Transformer 中注意力机制的作用"' in first


def test_ask_signature_changes_when_parameters_change() -> None:
    base = build_ask_signature(
        query="q",
        mode="hybrid",
        scope="all",
        prefer="lecture",
        per_file_cap=2,
        top_k=6,
        llm_model="qwen3:8b",
        evidence_policy="warn",
        detail="standard",
    )
    changed = build_ask_signature(
        query="q",
        mode="hybrid",
        scope="all",
        prefer="lecture",
        per_file_cap=2,
        top_k=8,
        llm_model="qwen3:8b",
        evidence_policy="warn",
        detail="standard",
        priority_answer_bank=True,
    )

    assert base != changed


def test_search_signature_changes_when_parameters_change() -> None:
    base = build_search_signature(query="q", mode="hybrid", scope="all", prefer="none", per_file_cap=0, top_k=10)
    changed = build_search_signature(query="q", mode="keyword", scope="all", prefer="none", per_file_cap=0, top_k=10)
    priority_changed = build_search_signature(
        query="q",
        mode="hybrid",
        scope="all",
        prefer="none",
        per_file_cap=0,
        top_k=10,
        priority_answer_bank=True,
    )

    assert base != changed
    assert base != priority_changed
    assert build_search_signature(query="q", mode="hybrid", scope="all", prefer="none", per_file_cap=0, top_k=10) == base


def test_preview_state_key_is_result_specific() -> None:
    first = preview_state_key("search", 1, "/tmp/a.pdf", 37)
    second = preview_state_key("search", 2, "/tmp/a.pdf", 37)
    third = preview_state_key("ask", 1, "/tmp/a.pdf", 37)

    assert first != second
    assert first != third
    assert "preview" in first


def test_preview_toggle_label() -> None:
    assert preview_toggle_label(False) == "预览该页"
    assert preview_toggle_label(True) == "隐藏预览"


def test_compact_status_text() -> None:
    assert compact_index_status(True, 11, 1379, True, 1379) == "Index: ready | docs 11 | chunks 1379 | semantic ready | embeddings 1379"
    assert compact_source_status(10, 1, 0) == "Source: lecture 10 | textbook_ocr 1 | other 0"
    assert compact_ollama_status(True, ["bge-m3", "qwen3:8b"]) == "Ollama: running | Models: bge-m3, qwen3:8b"
    assert compact_ollama_status(False, []) == "Ollama: not reachable"
