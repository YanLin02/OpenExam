from __future__ import annotations

import urllib.error
from io import BytesIO

from openexam.ask import (
    LLMError,
    NO_EVIDENCE,
    build_ask_prompt,
    context_supports_question,
    extract_question_key_phrases,
    format_evidence,
    format_source,
    ask_question,
    ollama_chat,
    render_ask_response,
)
from openexam.config import AppConfig
from openexam.models import SearchResult


def make_result(index: int = 1) -> SearchResult:
    return SearchResult(
        chunk_db_id=index,
        document_id=10,
        file_name="Chapter+4-Transformer.pdf",
        source_path="/tmp/Chapter+4-Transformer.pdf",
        source_type="lecture",
        location_type="page",
        location_label="p.37",
        page_number=37,
        chunk_id=f"chunk-{index}",
        text="Transformer 中的自注意力机制会计算查询、键和值之间的关系。",
        snippet="**Transformer** 中的自注意力机制会计算查询、键和值之间的关系。",
        score=90.0,
        mode="hybrid",
    )


def test_build_ask_prompt_limits_llm_to_chunks() -> None:
    prompt = build_ask_prompt("注意力机制的作用是什么？", [make_result()])

    assert "只能基于下面给出的检索片段回答" in prompt
    assert "不允许编造" in prompt
    assert NO_EVIDENCE in prompt
    assert "文件名: Chapter+4-Transformer.pdf" in prompt
    assert "路径: /tmp/Chapter+4-Transformer.pdf" in prompt
    assert "不要输出“回答：”“依据：”“来源：”标题" in prompt


def test_citation_formatting() -> None:
    result = make_result()

    assert format_evidence(result, 1) == "[1] Transformer 中的自注意力机制会计算查询、键和值之间的关系。"
    assert format_source(result, 1) == "[1] Chapter+4-Transformer.pdf, page 37, lecture, /tmp/Chapter+4-Transformer.pdf"


def test_no_results_do_not_call_llm(monkeypatch, tmp_path) -> None:
    called = False

    def fake_search(*args, **kwargs):
        return []

    def fake_chat(*args, **kwargs):
        nonlocal called
        called = True
        return "should not be called"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question("不存在的问题", config=AppConfig(index_dir=tmp_path / ".openexam"))

    assert response.answer == NO_EVIDENCE
    assert not response.llm_called
    assert not called
    assert render_ask_response(response) == NO_EVIDENCE


def test_ask_calls_search_then_llm(monkeypatch, tmp_path) -> None:
    def fake_search(*args, **kwargs):
        return [make_result()]

    def fake_chat(prompt, config, model=None):
        assert "检索片段" in prompt
        return "自注意力机制用于建模输入不同部分之间的相关性。[1]"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question("注意力机制的作用是什么？", config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_ask_response(response)

    assert response.llm_called
    assert "回答：" in rendered
    assert "依据：" in rendered
    assert "来源：" in rendered
    assert "Chapter+4-Transformer.pdf, page 37, lecture" in rendered


def test_insufficient_context_does_not_call_llm(monkeypatch, tmp_path) -> None:
    called = False

    def fake_search(*args, **kwargs):
        return [make_result()]

    def fake_chat(*args, **kwargs):
        nonlocal called
        called = True
        return "should not be called"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question(
        "卷积神经网络的局部连接和权值共享是什么意思",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
    )

    assert response.answer == NO_EVIDENCE
    assert not response.llm_called
    assert not called


def test_question_key_phrase_support() -> None:
    phrases = extract_question_key_phrases("为什么正则化可以缓解过拟合")
    assert phrases == ["正则化", "过拟合"]
    assert context_supports_question("Transformer 中注意力机制的作用", [make_result()])
    assert not context_supports_question("卷积神经网络的局部连接和权值共享是什么意思", [make_result()])


def test_ollama_unavailable_error(monkeypatch, tmp_path) -> None:
    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    try:
        ollama_chat("prompt", config=AppConfig(index_dir=tmp_path / ".openexam"))
    except LLMError as exc:
        assert str(exc) == "Ollama is not reachable. Start it with: ollama serve"
    else:
        raise AssertionError("Expected LLMError")


def test_ollama_missing_model_error(monkeypatch, tmp_path) -> None:
    class FakeHTTPError(urllib.error.HTTPError):
        def read(self):
            return b'{"error":"model not found"}'

    def fake_urlopen(*args, **kwargs):
        raise FakeHTTPError(url="http://127.0.0.1:11434/api/chat", code=404, msg="not found", hdrs=None, fp=BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    try:
        ollama_chat("prompt", config=AppConfig(index_dir=tmp_path / ".openexam"))
    except LLMError as exc:
        assert str(exc) == "Model qwen3:8b is missing. Pull it while online: ollama pull qwen3:8b"
    else:
        raise AssertionError("Expected LLMError")
