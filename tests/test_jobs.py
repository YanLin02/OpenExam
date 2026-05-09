from __future__ import annotations

import time

from openexam.ask import AskResponse
from openexam.jobs import (
    JobRecord,
    SearchJobResult,
    create_ask_executor,
    create_search_executor,
    job_elapsed_seconds,
    make_job_id,
    submit_ask_job,
    submit_search_job,
    update_job_from_future,
)


def wait_for_job(job: JobRecord, timeout: float = 2.0) -> JobRecord:
    deadline = time.time() + timeout
    while time.time() < deadline:
        update_job_from_future(job)
        if job.status in {"done", "error"}:
            return job
        time.sleep(0.01)
    update_job_from_future(job)
    return job


def test_job_record_initial_status() -> None:
    job = JobRecord(job_id=make_job_id(), kind="search", input_text="Transformer", signature="sig")

    assert job.status == "queued"
    assert job.result is None
    assert job.error is None
    assert isinstance(job_elapsed_seconds(job), float)


def test_submit_search_job_with_fake_search() -> None:
    def fake_search(query, *, timing, **kwargs):
        timing["total_time_ms"] = 1.0
        return []

    executor = create_search_executor(max_workers=1)
    try:
        job = submit_search_job(executor, "Transformer", signature="sig", search_fn=fake_search)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "done"
    assert isinstance(job.result, SearchJobResult)
    assert job.result.results == []
    assert job.result.timing["total_time_ms"] == 1.0
    assert job.error is None


def test_submit_ask_job_with_fake_ask() -> None:
    def fake_ask(question, *, mode, scope, prefer, per_file_cap, top_k, llm_model, evidence_policy, detail, **kwargs):
        return AskResponse(
            question=question,
            answer="answer",
            results=[],
            search_mode=mode,
            scope=scope,
            prefer=prefer,
            per_file_cap=per_file_cap,
            top_k=top_k,
            llm_model=llm_model,
            llm_called=False,
            evidence_status="none",
            evidence_policy=evidence_policy,
            missing_phrases=[],
            timing={"total_time_ms": 1.0},
            detail=detail,
        )

    executor = create_ask_executor(max_workers=1)
    try:
        job = submit_ask_job(executor, "为什么正则化可以缓解过拟合", signature="sig", ask_fn=fake_ask)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "done"
    assert isinstance(job.result, AskResponse)
    assert job.result.answer == "answer"
    assert job.error is None


def test_error_job_saves_error_message() -> None:
    def broken_search(*args, **kwargs):
        raise RuntimeError("search failed")

    executor = create_search_executor(max_workers=1)
    try:
        job = submit_search_job(executor, "bad query", signature="sig", search_fn=broken_search)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "error"
    assert job.result is None
    assert job.error is not None
    assert "search failed" in job.error


def test_executor_max_workers_defaults_and_overrides() -> None:
    search_executor = create_search_executor()
    ask_executor = create_ask_executor(max_workers=2)
    try:
        assert getattr(search_executor, "_max_workers") == 4
        assert getattr(ask_executor, "_max_workers") == 2
    finally:
        search_executor.shutdown(wait=True)
        ask_executor.shutdown(wait=True)
