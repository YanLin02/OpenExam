from __future__ import annotations

from openexam.ui_state import build_ask_signature


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
