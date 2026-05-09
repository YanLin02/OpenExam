from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.embeddings import embedding_status
from openexam.models import SearchResult
from openexam.ollama_utils import ensure_ollama_running
from openexam.search import SearchMode, search_index
from openexam.sources import SearchScope, SourcePreference
from openexam.text_utils import normalize_text


NO_EVIDENCE = "本地资料中未找到充分依据。"
PARTIAL_EVIDENCE_WARNING = "【资料依据不足】本地检索结果未充分覆盖问题中的全部关键点，以下回答包含模型基于通用知识的补充解释，请谨慎使用。"
NO_LOCAL_EVIDENCE_WARNING = "【未找到本地依据】本地资料中未检索到相关片段，以下回答主要来自模型通用知识，请谨慎使用。"
EvidenceStatus = Literal["sufficient", "partial", "none"]
EvidencePolicy = Literal["strict", "warn", "open"]
AnswerDetail = Literal["concise", "standard", "detailed"]
ASK_STOPWORDS = (
    "为什么",
    "是什么",
    "什么意思",
    "意思",
    "作用",
    "可以",
    "能够",
    "如何",
    "怎么",
    "请",
    "简述",
    "说明",
    "解释",
    "缓解",
    "的",
    "和",
    "与",
    "及",
    "中",
    "在",
)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class AskResponse:
    question: str
    answer: str
    results: list[SearchResult]
    search_mode: str
    scope: str
    prefer: str
    per_file_cap: int
    top_k: int
    llm_model: str
    llm_called: bool
    evidence_status: EvidenceStatus
    evidence_policy: EvidencePolicy
    missing_phrases: list[str]
    timing: dict[str, float]
    detail: AnswerDetail


def format_location(result: SearchResult) -> str:
    if result.page_number is not None:
        return f"page {result.page_number}"
    if result.slide_number is not None:
        return f"slide {result.slide_number}"
    if result.paragraph_index is not None:
        return f"paragraph {result.paragraph_index}"
    return result.location_label


def format_source(result: SearchResult, index: int) -> str:
    return f"[{index}] {result.file_name}, {format_location(result)}, {result.source_type}, {result.source_path}"


def format_evidence(result: SearchResult, index: int) -> str:
    snippet = result.snippet.replace("**", "")
    return f"[{index}] {snippet}"


def build_ask_prompt(
    question: str,
    results: list[SearchResult],
    evidence_status: EvidenceStatus = "sufficient",
    missing_phrases: list[str] | None = None,
    detail: AnswerDetail = "standard",
) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}]",
                    f"文件名: {result.file_name}",
                    f"位置: {format_location(result)}",
                    f"source_type: {result.source_type}",
                    f"路径: {result.source_path}",
                    f"原文片段: {result.text}",
                ]
            )
        )
    context = "\n\n".join(context_blocks)
    missing = "、".join(missing_phrases or []) or "无"
    status_rules = {
        "sufficient": "本地检索片段基本覆盖问题。请主要基于本地资料回答，并在回答中使用引用编号，例如 [1]、[2]。",
        "partial": "本地检索片段只覆盖了问题的一部分。请先说明本地资料支持了哪些点、缺少哪些点，再给出简短补充解释。补充解释必须明确是通用知识，不得伪装成本地资料。",
        "none": "没有本地检索片段。请明确说明没有本地依据，不得伪造来源、页码、文件名；可以基于通用知识给出简短解释。",
    }
    detail_rules = {
        "concise": "回答控制在 3-5 句话。只保留最关键解释，引用 2-3 个最相关片段即可。",
        "standard": "回答长度适中，覆盖问题主要方面，并保留必要引用。",
        "detailed": "可以分点说明并展开解释，但不得为了详细而编造来源；所有本地资料结论都必须保留引用。",
    }
    return f"""你是一个离线开卷考试资料检索助手。

资料依据状态：{evidence_status}
缺失关键点：{missing}
输出详细程度：{detail}

严格规则：
1. {status_rules[evidence_status]}
2. 不允许编造；所有情况下都禁止编造本地来源、页码、文件名。
3. 如果使用本地检索片段，必须包含引用编号，例如 [1]、[2]。
4. 如果没有本地检索片段，不要输出引用编号。
5. 只输出“回答”正文，不要输出“回答：”“依据：”“来源：”标题，也不要输出“补充说明：”标题。
6. 不要输出思考过程。
7. {detail_rules[detail]}

问题：
{question}

检索片段：
{context}
"""


def extract_question_key_phrases(question: str) -> list[str]:
    normalized = normalize_text(question)
    for word in ASK_STOPWORDS:
        normalized = normalized.replace(word, " ")
    phrases = [part.strip() for part in normalized.split() if len(part.strip()) >= 2]
    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in phrases:
        if phrase not in seen:
            ordered.append(phrase)
            seen.add(phrase)
    return ordered


def _phrase_supported(phrase: str, context_norm: str) -> bool:
    if phrase in context_norm:
        return True
    if len(phrase) < 3:
        return False
    grams = [phrase[index : index + 2] for index in range(len(phrase) - 1)]
    matched = sum(1 for gram in grams if gram in context_norm)
    return matched / len(grams) >= 0.5


def context_supports_question(question: str, results: list[SearchResult]) -> bool:
    return evidence_status_for_question(question, results)[0] == "sufficient"


def evidence_status_for_question(question: str, results: list[SearchResult]) -> tuple[EvidenceStatus, list[str]]:
    if not results:
        return "none", []
    phrases = extract_question_key_phrases(question)
    if not phrases:
        return "sufficient", []
    context_norm = normalize_text(" ".join(result.text for result in results))
    missing = [phrase for phrase in phrases if not _phrase_supported(phrase, context_norm)]
    if missing:
        return "partial", missing
    return "sufficient", []


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _normalize_answer_text(text: str) -> str:
    text = _strip_thinking(text)
    text = re.split(r"\n\s*(依据|来源)\s*[:：]", text, maxsplit=1)[0].strip()
    text = re.sub(r"^回答\s*[:：]\s*", "", text).strip()
    return text


def _post_json(url: str, payload: dict[str, Any], timeout: float, model: str) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 or "not found" in body.lower() or "pull" in body.lower():
            raise LLMError(f"Model {model} is missing. Pull it while online: ollama pull {model}") from exc
        raise LLMError(f"Ollama LLM request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError("Ollama is not reachable. Start it with: ollama serve") from exc
    except TimeoutError as exc:
        raise LLMError("Ollama LLM request timed out.") from exc


def num_predict_for_detail(detail: AnswerDetail) -> int:
    return {"concise": 512, "standard": 1024, "detailed": 2048}[detail]


def ollama_chat(prompt: str, config: AppConfig = DEFAULT_CONFIG, model: str | None = None, num_predict: int = 1024) -> str:
    if config.llm_provider != "ollama":
        raise LLMError(f"Unsupported LLM provider: {config.llm_provider}")
    llm_model = model or config.llm_model
    url = config.ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": llm_model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": "你必须遵守用户提示中的资料依据状态。不得编造本地来源、页码或文件名。不要输出思考过程。",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    response = _post_json(url, payload, timeout=config.llm_timeout_seconds, model=llm_model)
    message = response.get("message", {})
    content = message.get("content")
    if not content:
        raise LLMError("Ollama LLM response did not include content.")
    return _normalize_answer_text(content)


def render_ask_response(response: AskResponse) -> str:
    if response.answer.strip() == NO_EVIDENCE:
        return NO_EVIDENCE
    if response.evidence_status == "none":
        answer = response.answer.strip()
        if not answer.startswith(NO_LOCAL_EVIDENCE_WARNING):
            answer = f"{NO_LOCAL_EVIDENCE_WARNING}\n{answer}"
        return (
            f"回答：\n{answer}\n\n"
            "资料依据状态：\nnone\n\n"
            "依据：\n无本地依据\n\n"
            "来源：\n无本地来源\n\n"
            "补充说明：\n模型回答主要来自通用知识，考试使用时请谨慎核对。"
        )
    evidence = "\n".join(format_evidence(result, index) for index, result in enumerate(response.results, start=1))
    sources = "\n".join(format_source(result, index) for index, result in enumerate(response.results, start=1))
    answer = response.answer.strip() or NO_EVIDENCE
    if response.evidence_status == "partial":
        if not answer.startswith(PARTIAL_EVIDENCE_WARNING):
            answer = f"{PARTIAL_EVIDENCE_WARNING}\n{answer}"
        missing = "、".join(response.missing_phrases) or "未完全覆盖"
        return (
            f"回答：\n{answer}\n\n"
            f"资料依据状态：\npartial，缺失关键点：{missing}\n\n"
            f"依据：\n{evidence}\n\n"
            f"来源：\n{sources}\n\n"
            "补充说明：\n本回答包含模型基于通用知识的补充解释；考试使用时请优先核对上方本地来源。"
        )
    return f"回答：\n{answer}\n\n依据：\n{evidence}\n\n来源：\n{sources}"


def ask_question(
    question: str,
    config: AppConfig = DEFAULT_CONFIG,
    mode: SearchMode = "hybrid",
    scope: SearchScope = "all",
    prefer: SourcePreference = "lecture",
    per_file_cap: int = 2,
    top_k: int | None = None,
    llm_model: str | None = None,
    evidence_policy: EvidencePolicy = "warn",
    detail: AnswerDetail = "standard",
    auto_start_ollama: bool = True,
) -> AskResponse:
    total_start = time.perf_counter()
    effective_top_k = top_k if top_k is not None else config.llm_context_top_k
    effective_mode = mode
    if mode == "semantic" and not embedding_status(config).valid:
        effective_mode = "hybrid"
    search_timing: dict[str, float] = {}
    retrieval_start = time.perf_counter()
    results = search_index(
        question,
        top_k=effective_top_k,
        config=config,
        mode=effective_mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        timing=search_timing,
    )
    retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
    evidence_status, missing_phrases = evidence_status_for_question(question, results)
    if evidence_policy == "strict" and evidence_status != "sufficient":
        return AskResponse(
            question=question,
            answer=NO_EVIDENCE,
            results=results,
            search_mode=effective_mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=per_file_cap,
            top_k=effective_top_k,
            llm_model=llm_model or config.llm_model,
            llm_called=False,
            evidence_status=evidence_status,
            evidence_policy=evidence_policy,
            missing_phrases=missing_phrases,
            timing={
                "retrieval_time_ms": retrieval_time_ms,
                "prompt_build_time_ms": 0.0,
                "llm_time_ms": 0.0,
                "total_time_ms": (time.perf_counter() - total_start) * 1000,
            },
            detail=detail,
        )
    prompt_start = time.perf_counter()
    prompt = build_ask_prompt(question, results, evidence_status=evidence_status, missing_phrases=missing_phrases, detail=detail)
    prompt_build_time_ms = (time.perf_counter() - prompt_start) * 1000
    ollama_status = ensure_ollama_running(config.ollama_base_url, auto_start=auto_start_ollama, log_path=config.index_dir / "ollama.log")
    if not ollama_status.reachable:
        raise LLMError(ollama_status.message)
    llm_start = time.perf_counter()
    answer = ollama_chat(prompt, config=config, model=llm_model, num_predict=num_predict_for_detail(detail))
    llm_time_ms = (time.perf_counter() - llm_start) * 1000
    return AskResponse(
        question=question,
        answer=answer,
        results=results,
        search_mode=effective_mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=effective_top_k,
        llm_model=llm_model or config.llm_model,
        llm_called=True,
        evidence_status=evidence_status,
        evidence_policy=evidence_policy,
        missing_phrases=missing_phrases,
        timing={
            "retrieval_time_ms": retrieval_time_ms,
            "prompt_build_time_ms": prompt_build_time_ms,
            "llm_time_ms": llm_time_ms,
            "total_time_ms": (time.perf_counter() - total_start) * 1000,
        },
        detail=detail,
    )
