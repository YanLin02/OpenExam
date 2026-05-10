from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from openexam.ask import AnswerDetail, AskResponse, EvidencePolicy, ask_question
from openexam.config import AppConfig, DEFAULT_CONFIG
from openexam.models import SearchResult
from openexam.search import SearchMode, search_index
from openexam.sources import SearchScope, SourcePreference


JobStatus = Literal["queued", "running", "done", "error"]
JobKind = Literal["search", "ask"]


@dataclass
class SearchJobResult:
    results: list[SearchResult]
    timing: dict[str, float]


@dataclass
class JobRecord:
    job_id: str
    kind: JobKind
    input_text: str
    signature: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: object | None = None
    error: str | None = None
    future: Future[Any] | None = field(default=None, repr=False, compare=False)


SearchFunction = Callable[..., list[SearchResult]]
AskFunction = Callable[..., AskResponse]


def make_job_id() -> str:
    return uuid.uuid4().hex


def create_search_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openexam-search")


def create_ask_executor(max_workers: int = 1) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="openexam-ask")


def _run_search_job(
    query: str,
    *,
    config: AppConfig,
    mode: SearchMode,
    scope: SearchScope,
    prefer: SourcePreference,
    per_file_cap: int,
    top_k: int,
    search_fn: SearchFunction,
) -> SearchJobResult:
    timing: dict[str, float] = {}
    results = search_fn(
        query,
        top_k=top_k,
        config=config,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        timing=timing,
    )
    return SearchJobResult(results=results, timing=timing)


def submit_search_job(
    executor: ThreadPoolExecutor,
    query: str,
    *,
    signature: str,
    config: AppConfig = DEFAULT_CONFIG,
    mode: SearchMode = "hybrid",
    scope: SearchScope = "all",
    prefer: SourcePreference = "none",
    per_file_cap: int = 0,
    top_k: int = 10,
    search_fn: SearchFunction = search_index,
) -> JobRecord:
    job = JobRecord(job_id=make_job_id(), kind="search", input_text=query, signature=signature)
    job.future = executor.submit(
        _run_search_job,
        query,
        config=config,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=top_k,
        search_fn=search_fn,
    )
    return job


def _run_ask_job(
    question: str,
    *,
    config: AppConfig,
    mode: SearchMode,
    scope: SearchScope,
    prefer: SourcePreference,
    per_file_cap: int,
    top_k: int,
    llm_model: str,
    evidence_policy: EvidencePolicy,
    detail: AnswerDetail,
    ask_fn: AskFunction,
) -> AskResponse:
    return ask_fn(
        question,
        config=config,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=top_k,
        llm_model=llm_model,
        evidence_policy=evidence_policy,
        detail=detail,
    )


def submit_ask_job(
    executor: ThreadPoolExecutor,
    question: str,
    *,
    signature: str,
    config: AppConfig = DEFAULT_CONFIG,
    mode: SearchMode = "hybrid",
    scope: SearchScope = "all",
    prefer: SourcePreference = "lecture",
    per_file_cap: int = 2,
    top_k: int | None = None,
    llm_model: str = DEFAULT_CONFIG.llm_model,
    evidence_policy: EvidencePolicy = "warn",
    detail: AnswerDetail = "standard",
    ask_fn: AskFunction = ask_question,
) -> JobRecord:
    effective_top_k = top_k if top_k is not None else config.llm_context_top_k
    job = JobRecord(job_id=make_job_id(), kind="ask", input_text=question, signature=signature)
    job.future = executor.submit(
        _run_ask_job,
        question,
        config=config,
        mode=mode,
        scope=scope,
        prefer=prefer,
        per_file_cap=per_file_cap,
        top_k=effective_top_k,
        llm_model=llm_model,
        evidence_policy=evidence_policy,
        detail=detail,
        ask_fn=ask_fn,
    )
    return job


def update_job_from_future(job: JobRecord) -> JobRecord:
    future = job.future
    if future is None or job.status in {"done", "error"}:
        return job

    now = time.time()
    if future.running():
        job.status = "running"
        if job.started_at is None:
            job.started_at = now
        return job

    if not future.done():
        return job

    if job.started_at is None:
        job.started_at = now
    job.finished_at = now
    try:
        job.result = future.result()
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.result = None
    else:
        job.status = "done"
        job.error = None
    return job


def job_elapsed_seconds(job: JobRecord) -> float:
    end = job.finished_at if job.finished_at is not None else time.time()
    return max(0.0, end - job.created_at)


def job_preview_prefix(kind: JobKind, job_id: str) -> str:
    return f"{kind}-job-{job_id}"


def is_job_collapsed(collapsed_ids: set[str], job_id: str) -> bool:
    return job_id in collapsed_ids


def toggle_job_collapsed(collapsed_ids: set[str], job_id: str) -> set[str]:
    updated = set(collapsed_ids)
    if job_id in updated:
        updated.remove(job_id)
    else:
        updated.add(job_id)
    return updated


def remove_job_by_id(jobs: list[JobRecord], job_id: str) -> list[JobRecord]:
    return [job for job in jobs if job.job_id != job_id]


def close_job_by_id(jobs: list[JobRecord], job_id: str) -> list[JobRecord]:
    for job in jobs:
        if job.job_id == job_id and job.status == "queued" and job.future is not None:
            job.future.cancel()
            break
    return remove_job_by_id(jobs, job_id)
