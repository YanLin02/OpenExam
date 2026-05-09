from __future__ import annotations

from openexam.ui_state import build_ask_signature, build_search_signature, preview_state_key


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
    )

    assert base != changed


def test_search_signature_changes_when_parameters_change() -> None:
    base = build_search_signature(query="q", mode="hybrid", scope="all", prefer="none", per_file_cap=0, top_k=10)
    changed = build_search_signature(query="q", mode="keyword", scope="all", prefer="none", per_file_cap=0, top_k=10)

    assert base != changed
    assert build_search_signature(query="q", mode="hybrid", scope="all", prefer="none", per_file_cap=0, top_k=10) == base


def test_preview_state_key_is_result_specific() -> None:
    first = preview_state_key("search", 1, "/tmp/a.pdf", 37)
    second = preview_state_key("search", 2, "/tmp/a.pdf", 37)
    third = preview_state_key("ask", 1, "/tmp/a.pdf", 37)

    assert first != second
    assert first != third
    assert "preview" in first
