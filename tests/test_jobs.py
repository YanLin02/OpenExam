from __future__ import annotations

import time
from concurrent.futures import Future

from openexam.ask import AskResponse
from openexam.jobs import (
    JobRecord,
    SearchJobResult,
    close_job_by_id,
    create_ask_executor,
    create_search_executor,
    create_solve_executor,
    is_job_collapsed,
    job_elapsed_seconds,
    job_preview_prefix,
    jobs_in_submission_order,
    make_job_id,
    queue_input_key,
    remove_job_by_id,
    submit_ask_job,
    submit_search_job,
    submit_solve_job,
    toggle_job_collapsed,
    update_job_from_future,
)
from openexam.problem_types import ProblemType
from openexam.solve import SolveResponse


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


def test_job_kind_supports_solve() -> None:
    job = JobRecord(job_id=make_job_id(), kind="solve", input_text="设计一个 CNN", signature="sig")

    assert job.kind == "solve"
    assert job_preview_prefix("solve", "abc123") == "solve-job-abc123"
    assert queue_input_key("solve") == "solve_queue_input"


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


def test_submit_ask_job_can_use_solve_mode() -> None:
    def fake_solve(question, *, mode, search_mode, scope, prefer, per_file_cap, top_k, llm_model, evidence_policy, detail, **kwargs):
        ask_response = AskResponse(
            question=question,
            answer="solve answer",
            results=[],
            search_mode=search_mode,
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
        return SolveResponse(
            question=question,
            problem_type=ProblemType.CALCULATION,
            strategy="先识别已知量。",
            ask_response=ask_response,
            requested_mode=mode,
        )

    executor = create_ask_executor(max_workers=1)
    try:
        job = submit_ask_job(executor, "输出尺寸是多少", signature="sig", answer_mode="solve", solve_fn=fake_solve)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "done"
    assert isinstance(job.result, SolveResponse)
    assert job.result.problem_type == ProblemType.CALCULATION
    assert job.error is None


def test_submit_solve_job_with_fake_solve() -> None:
    def fake_solve(question, *, mode, search_mode, scope, prefer, per_file_cap, top_k, llm_model, evidence_policy, detail, **kwargs):
        ask_response = AskResponse(
            question=question,
            answer="solve answer",
            results=[],
            search_mode=search_mode,
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
        return SolveResponse(
            question=question,
            problem_type=ProblemType.DESIGN,
            strategy="按模板作答。",
            ask_response=ask_response,
            requested_mode=mode,
        )

    executor = create_solve_executor(max_workers=1)
    try:
        job = submit_solve_job(executor, "设计一个 CNN", signature="sig", problem_type="design", solve_fn=fake_solve)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "done"
    assert isinstance(job.result, SolveResponse)
    assert job.result.problem_type == ProblemType.DESIGN
    assert job.error is None


def test_solve_job_error_saves_error_message() -> None:
    def broken_solve(*args, **kwargs):
        raise RuntimeError("solve failed")

    executor = create_solve_executor(max_workers=1)
    try:
        job = submit_solve_job(executor, "bad problem", signature="sig", solve_fn=broken_solve)
        wait_for_job(job)
    finally:
        executor.shutdown(wait=True)

    assert job.status == "error"
    assert job.result is None
    assert job.error is not None
    assert "solve failed" in job.error


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
    solve_executor = create_solve_executor()
    try:
        assert getattr(search_executor, "_max_workers") == 4
        assert getattr(ask_executor, "_max_workers") == 2
        assert getattr(solve_executor, "_max_workers") == 1
    finally:
        search_executor.shutdown(wait=True)
        ask_executor.shutdown(wait=True)
        solve_executor.shutdown(wait=True)


def test_job_collapsed_state_helpers() -> None:
    job_id = "abc123"
    collapsed: set[str] = set()

    assert job_preview_prefix("ask", job_id) == "ask-job-abc123"
    assert not is_job_collapsed(collapsed, job_id)

    collapsed = toggle_job_collapsed(collapsed, job_id)
    assert is_job_collapsed(collapsed, job_id)

    collapsed = toggle_job_collapsed(collapsed, job_id)
    assert not is_job_collapsed(collapsed, job_id)


def test_queue_input_keys_are_mode_specific() -> None:
    assert queue_input_key("search") == "search_queue_input"
    assert queue_input_key("ask") == "ask_queue_input"
    assert queue_input_key("solve") == "solve_queue_input"
    assert queue_input_key("search") != queue_input_key("ask")
    assert queue_input_key("solve") != queue_input_key("ask")


def test_jobs_in_submission_order_does_not_reverse() -> None:
    first = JobRecord(job_id="first", kind="search", input_text="first query", signature="sig1")
    second = JobRecord(job_id="second", kind="search", input_text="second query", signature="sig2")

    ordered = jobs_in_submission_order([first, second])

    assert ordered == [first, second]


def test_remove_job_by_id_removes_only_target() -> None:
    first = JobRecord(job_id="first", kind="search", input_text="q1", signature="sig1")
    second = JobRecord(job_id="second", kind="ask", input_text="q2", signature="sig2")
    third = JobRecord(job_id="third", kind="solve", input_text="q3", signature="sig3")

    remaining = remove_job_by_id([first, second, third], "third")

    assert remaining == [first, second]


def test_solve_queue_state_helpers_do_not_affect_search_or_ask_jobs() -> None:
    search_job = JobRecord(job_id="search", kind="search", input_text="q1", signature="sig1")
    ask_job = JobRecord(job_id="ask", kind="ask", input_text="q2", signature="sig2")
    solve_job = JobRecord(job_id="solve", kind="solve", input_text="q3", signature="sig3")
    collapsed = toggle_job_collapsed(set(), solve_job.job_id)

    assert is_job_collapsed(collapsed, "solve")
    assert not is_job_collapsed(collapsed, "search")
    assert not is_job_collapsed(collapsed, "ask")
    assert remove_job_by_id([search_job, ask_job], solve_job.job_id) == [search_job, ask_job]


def test_close_queued_job_cancels_future_and_removes_job() -> None:
    future: Future[object] = Future()
    job = JobRecord(job_id="queued", kind="ask", input_text="q", signature="sig", future=future)

    remaining = close_job_by_id([job], "queued")

    assert remaining == []
    assert future.cancelled()


def test_close_running_job_removes_without_force_stopping() -> None:
    future: Future[object] = Future()
    future.set_running_or_notify_cancel()
    job = JobRecord(job_id="running", kind="ask", input_text="q", signature="sig", status="running", future=future)

    remaining = close_job_by_id([job], "running")

    assert remaining == []
    assert future.running()
