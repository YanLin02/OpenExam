from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openexam.ask import AskResponse
from openexam.config import AppConfig
from openexam.problem_types import ProblemType, classify_problem
from openexam.solve import render_solve_response, solve_question


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "exam_questions.json"


def load_exam_questions() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


EXAM_QUESTIONS = load_exam_questions()


def exam_question_id(case: dict[str, Any]) -> str:
    return str(case["id"])


def fake_ask_response(question: str, *, llm_called: bool = True, **kwargs: Any) -> AskResponse:
    return AskResponse(
        question=question,
        answer="fake local answer with template sections",
        results=[],
        search_mode=kwargs.get("mode", "hybrid"),
        scope=kwargs.get("scope", "all"),
        prefer=kwargs.get("prefer", "lecture"),
        per_file_cap=kwargs.get("per_file_cap", 2),
        top_k=kwargs.get("top_k") or 6,
        llm_model=kwargs.get("llm_model") or "qwen3:8b",
        llm_called=llm_called,
        evidence_status="sufficient",
        evidence_policy=kwargs.get("evidence_policy", "warn"),
        missing_phrases=[],
        timing={"retrieval_time_ms": 0.0, "prompt_build_time_ms": 0.0, "llm_time_ms": 1.0, "total_time_ms": 1.0},
        detail=kwargs.get("detail", "standard"),
    )


@pytest.mark.parametrize("case", EXAM_QUESTIONS, ids=exam_question_id)
def test_classify_problem_matches_exam_fixture(case: dict[str, Any]) -> None:
    assert classify_problem(case["question"]) == ProblemType(case["expected_problem_type"])


@pytest.mark.parametrize(
    "case",
    [case for case in EXAM_QUESTIONS if case["expected_problem_type"] == "calculation"],
    ids=exam_question_id,
)
def test_calculation_exam_fixtures_use_calculators(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: dict[str, Any]) -> None:
    def fail_if_llm_is_called(*args: Any, **kwargs: Any) -> AskResponse:
        raise AssertionError("calculation fixture unexpectedly called ask_question")

    if case["expected_llm_called"] is False:
        monkeypatch.setattr("openexam.solve.ask_question", fail_if_llm_is_called)
    else:
        monkeypatch.setattr("openexam.solve.ask_question", fake_ask_response)
    response = solve_question(case["question"], config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.CALCULATION
    assert response.calculation_answer is not None
    assert response.calculation_answer.calculation_type == case["expected_subtype"]
    assert response.ask_response.llm_called is case["expected_llm_called"]
    if case["expected_llm_called"] is False:
        assert not response.calculation_answer.need_manual_input
    else:
        assert response.calculation_answer.need_manual_input
    for expected in case["expected_contains"]:
        assert expected in rendered
    for unexpected in case["expected_not_contains"]:
        assert unexpected not in rendered


@pytest.mark.parametrize(
    "case",
    [case for case in EXAM_QUESTIONS if case["expected_problem_type"] == "design"],
    ids=exam_question_id,
)
def test_design_exam_fixtures_use_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: dict[str, Any]) -> None:
    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_response)
    response = solve_question(case["question"], config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.DESIGN
    assert response.design_task_type is not None
    assert response.design_task_type.value == case["expected_subtype"]
    assert response.ask_response.llm_called is True
    assert "答题结构" in rendered
    for expected in case["expected_contains"]:
        assert expected in rendered


@pytest.mark.parametrize(
    "case",
    [case for case in EXAM_QUESTIONS if case["expected_problem_type"] == "derivation"],
    ids=exam_question_id,
)
def test_derivation_exam_fixtures_use_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: dict[str, Any]) -> None:
    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_response)
    response = solve_question(case["question"], config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.DERIVATION
    assert response.derivation_task_type is not None
    assert response.derivation_task_type.value == case["expected_subtype"]
    assert response.ask_response.llm_called is True
    assert "推导结构" in rendered
    for expected in case["expected_contains"]:
        assert expected in rendered


@pytest.mark.parametrize(
    "case",
    [case for case in EXAM_QUESTIONS if case["expected_problem_type"] == "compare"],
    ids=exam_question_id,
)
def test_compare_exam_fixtures_use_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: dict[str, Any]) -> None:
    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_response)
    response = solve_question(case["question"], config=AppConfig(index_dir=tmp_path / ".openexam"))
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.COMPARE
    assert response.compare_task_type is not None
    assert response.compare_task_type.value == case["expected_subtype"]
    assert response.ask_response.llm_called is True
    assert "比较维度" in rendered
    for expected in case["expected_contains"]:
        assert expected in rendered


@pytest.mark.parametrize(
    "case",
    [case for case in EXAM_QUESTIONS if case["expected_problem_type"] in {"concept", "short_answer"}],
    ids=exam_question_id,
)
def test_concept_and_short_answer_exam_fixtures_only_classify(case: dict[str, Any]) -> None:
    assert classify_problem(case["question"]) == ProblemType(case["expected_problem_type"])
