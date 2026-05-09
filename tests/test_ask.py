from __future__ import annotations

import urllib.error
from io import BytesIO

from openexam.ask import (
    LLMError,
    NO_EVIDENCE,
    NO_LOCAL_EVIDENCE_WARNING,
    PARTIAL_EVIDENCE_WARNING,
    build_ask_prompt,
    context_supports_question,
    evidence_status_for_question,
    extract_question_key_phrases,
    format_evidence,
    format_source,
    ask_question,
    ollama_chat,
    num_predict_for_detail,
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


def fake_ollama_ready(*args, **kwargs):
    class Status:
        reachable = True
        message = "ok"

    return Status()


def test_build_ask_prompt_limits_llm_to_chunks() -> None:
    prompt = build_ask_prompt("注意力机制的作用是什么？", [make_result()])

    assert "资料依据状态：sufficient" in prompt
    assert "不允许编造" in prompt
    assert "文件名: Chapter+4-Transformer.pdf" in prompt
    assert "路径: /tmp/Chapter+4-Transformer.pdf" in prompt
    assert "不要输出“回答：”“依据：”“来源：”标题" in prompt


def test_citation_formatting() -> None:
    result = make_result()

    assert format_evidence(result, 1) == "[1] Transformer 中的自注意力机制会计算查询、键和值之间的关系。"
    assert format_source(result, 1) == "[1] Chapter+4-Transformer.pdf, page 37, lecture, /tmp/Chapter+4-Transformer.pdf"


def test_no_results_strict_does_not_call_llm(monkeypatch, tmp_path) -> None:
    called = False

    def fake_search(*args, **kwargs):
        return []

    def fake_chat(*args, **kwargs):
        nonlocal called
        called = True
        return "should not be called"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question("不存在的问题", config=AppConfig(index_dir=tmp_path / ".openexam"), evidence_policy="strict")

    assert response.answer == NO_EVIDENCE
    assert not response.llm_called
    assert not called
    assert render_ask_response(response) == NO_EVIDENCE


def test_no_results_warn_calls_llm_without_fake_sources(monkeypatch, tmp_path) -> None:
    called = False

    def fake_search(*args, **kwargs):
        return []

    def fake_chat(prompt, config, model=None, num_predict=1024):
        nonlocal called
        called = True
        assert "资料依据状态：none" in prompt
        return "没有本地依据时，只能给出通用解释。"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ensure_ollama_running", fake_ollama_ready)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question("一个本地资料中不存在的随机问题", config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_ask_response(response)

    assert called
    assert response.llm_called
    assert response.evidence_status == "none"
    assert NO_LOCAL_EVIDENCE_WARNING in rendered
    assert "依据：\n无本地依据" in rendered
    assert "来源：\n无本地来源" in rendered


def test_ask_calls_search_then_llm(monkeypatch, tmp_path) -> None:
    def fake_search(*args, **kwargs):
        return [make_result()]

    def fake_chat(prompt, config, model=None, num_predict=1024):
        assert "检索片段" in prompt
        return "自注意力机制用于建模输入不同部分之间的相关性。[1]"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ensure_ollama_running", fake_ollama_ready)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question("注意力机制的作用是什么？", config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_ask_response(response)

    assert response.llm_called
    assert "回答：" in rendered
    assert "依据：" in rendered
    assert "来源：" in rendered
    assert "Chapter+4-Transformer.pdf, page 37, lecture" in rendered


def test_partial_evidence_warn_calls_llm_with_warning(monkeypatch, tmp_path) -> None:
    called = False

    def fake_search(*args, **kwargs):
        return [make_result()]

    def fake_chat(*args, **kwargs):
        nonlocal called
        called = True
        assert "资料依据状态：partial" in args[0]
        return "本地资料只支持 Transformer 注意力部分；局部连接和权值共享需要通用知识补充。"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ensure_ollama_running", fake_ollama_ready)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question(
        "卷积神经网络的局部连接和权值共享是什么意思",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
    )
    rendered = render_ask_response(response)

    assert response.evidence_status == "partial"
    assert response.llm_called
    assert called
    assert PARTIAL_EVIDENCE_WARNING in rendered
    assert "资料依据状态：" in rendered
    assert "补充说明：" in rendered


def test_insufficient_context_strict_does_not_call_llm(monkeypatch, tmp_path) -> None:
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
        evidence_policy="strict",
    )

    assert response.answer == NO_EVIDENCE
    assert response.evidence_status == "partial"
    assert not response.llm_called
    assert not called


def test_open_policy_no_results_calls_llm(monkeypatch, tmp_path) -> None:
    def fake_search(*args, **kwargs):
        return []

    def fake_chat(prompt, config, model=None, num_predict=1024):
        assert "资料依据状态：none" in prompt
        return "这是通用知识解释。"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ensure_ollama_running", fake_ollama_ready)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)
    response = ask_question(
        "一个本地资料中不存在的随机问题",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        evidence_policy="open",
    )
    rendered = render_ask_response(response)

    assert response.llm_called
    assert response.evidence_status == "none"
    assert NO_LOCAL_EVIDENCE_WARNING in rendered
    assert "无本地来源" in rendered


def test_question_key_phrase_support() -> None:
    phrases = extract_question_key_phrases("为什么正则化可以缓解过拟合")
    assert phrases == ["正则化", "过拟合"]
    assert context_supports_question("Transformer 中注意力机制的作用", [make_result()])
    assert not context_supports_question("卷积神经网络的局部连接和权值共享是什么意思", [make_result()])
    assert evidence_status_for_question("卷积神经网络的局部连接和权值共享是什么意思", [make_result()])[0] == "partial"
    assert evidence_status_for_question("一个本地资料中不存在的随机问题", [])[0] == "none"


def test_detail_controls_prompt_and_num_predict(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_search(*args, **kwargs):
        return [make_result()]

    def fake_chat(prompt, config, model=None, num_predict=1024):
        seen["prompt"] = prompt
        seen["num_predict"] = num_predict
        return "简短回答。[1]"

    monkeypatch.setattr("openexam.ask.search_index", fake_search)
    monkeypatch.setattr("openexam.ask.ensure_ollama_running", fake_ollama_ready)
    monkeypatch.setattr("openexam.ask.ollama_chat", fake_chat)

    response = ask_question("注意力机制的作用是什么？", config=AppConfig(index_dir=tmp_path / ".openexam"), detail="concise")

    assert response.timing["retrieval_time_ms"] >= 0
    assert response.timing["llm_time_ms"] >= 0
    assert response.detail == "concise"
    assert "输出详细程度：concise" in str(seen["prompt"])
    assert seen["num_predict"] == 512
    assert num_predict_for_detail("detailed") == 2048


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
