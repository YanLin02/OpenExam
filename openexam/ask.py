from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.embeddings import embedding_status
from openexam.models import SearchResult
from openexam.search import SearchMode, search_index
from openexam.sources import SearchScope, SourcePreference
from openexam.text_utils import normalize_text


NO_EVIDENCE = "本地资料中未找到充分依据。"
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


def build_ask_prompt(question: str, results: list[SearchResult]) -> str:
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
    return f"""你是一个离线开卷考试资料检索助手。你只能基于下面给出的检索片段回答问题。

严格规则：
1. 只能使用“检索片段”中的信息回答。
2. 不允许编造检索片段之外的内容。
3. 如果检索片段不足以回答，必须只回答：{NO_EVIDENCE}
4. 回答必须包含引用编号，例如 [1]、[2]。
5. 只输出“回答”正文，不要输出“回答：”“依据：”“来源：”标题。
6. 不要输出思考过程，不要输出资料之外的推测。

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
    phrases = extract_question_key_phrases(question)
    if not phrases:
        return bool(results)
    context_norm = normalize_text(" ".join(result.text for result in results))
    return all(_phrase_supported(phrase, context_norm) for phrase in phrases)


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


def ollama_chat(prompt: str, config: AppConfig = DEFAULT_CONFIG, model: str | None = None) -> str:
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
                "content": "你只能基于用户提供的检索片段回答。不要编造，不要输出思考过程。",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0},
    }
    response = _post_json(url, payload, timeout=config.llm_timeout_seconds, model=llm_model)
    message = response.get("message", {})
    content = message.get("content")
    if not content:
        raise LLMError("Ollama LLM response did not include content.")
    return _normalize_answer_text(content)


def render_ask_response(response: AskResponse) -> str:
    if not response.results:
        return NO_EVIDENCE
    evidence = "\n".join(format_evidence(result, index) for index, result in enumerate(response.results, start=1))
    sources = "\n".join(format_source(result, index) for index, result in enumerate(response.results, start=1))
    answer = response.answer.strip() or NO_EVIDENCE
    if answer == NO_EVIDENCE:
        return NO_EVIDENCE
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
) -> AskResponse:
    effective_top_k = top_k if top_k is not None else config.llm_context_top_k
    effective_mode = mode
    if mode == "semantic" and not embedding_status(config).valid:
        effective_mode = "hybrid"
    results = search_index(
        question,
        top_k=effective_top_k,
        config=config,
        mode=effective_mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
    )
    if not results:
        return AskResponse(
            question=question,
            answer=NO_EVIDENCE,
            results=[],
            search_mode=effective_mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=per_file_cap,
            top_k=effective_top_k,
            llm_model=llm_model or config.llm_model,
            llm_called=False,
        )
    if not context_supports_question(question, results):
        return AskResponse(
            question=question,
            answer=NO_EVIDENCE,
            results=[],
            search_mode=effective_mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=per_file_cap,
            top_k=effective_top_k,
            llm_model=llm_model or config.llm_model,
            llm_called=False,
        )
    prompt = build_ask_prompt(question, results)
    answer = ollama_chat(prompt, config=config, model=llm_model)
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
    )
