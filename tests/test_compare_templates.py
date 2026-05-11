from __future__ import annotations

from openexam.ask import AskResponse
from openexam.compare_templates import (
    CompareTaskType,
    build_compare_answer_plan,
    build_compare_prompt,
    classify_compare_task,
)
from openexam.config import AppConfig
from openexam.problem_types import ProblemType
from openexam.solve import render_solve_response, solve_question


def make_ask_response(question: str) -> AskResponse:
    return AskResponse(
        question=question,
        answer="1. 比较对象：CNN 与 Transformer。\n6. 考试总结：按任务选择模型。",
        results=[],
        search_mode="hybrid",
        scope="all",
        prefer="lecture",
        per_file_cap=2,
        top_k=6,
        llm_model="qwen3:8b",
        llm_called=True,
        evidence_status="none",
        evidence_policy="warn",
        missing_phrases=[],
        timing={"retrieval_time_ms": 1.0, "prompt_build_time_ms": 1.0, "llm_time_ms": 1.0, "total_time_ms": 3.0},
        detail="standard",
    )


def test_classify_compare_task_examples() -> None:
    assert classify_compare_task("比较 CNN 和 Transformer 的区别") == CompareTaskType.MODEL_COMPARISON
    assert classify_compare_task("比较 SGD 和 Adam") == CompareTaskType.OPTIMIZER_COMPARISON
    assert classify_compare_task("比较 MSE 和交叉熵") == CompareTaskType.LOSS_COMPARISON
    assert classify_compare_task("比较 dropout 和 L2 正则化") == CompareTaskType.REGULARIZATION_COMPARISON
    assert classify_compare_task("比较 accuracy precision recall F1") == CompareTaskType.METRIC_COMPARISON


def test_build_compare_answer_plan_contains_sections_and_dimensions() -> None:
    plan = build_compare_answer_plan("比较 CNN 和 Transformer 的区别", CompareTaskType.MODEL_COMPARISON)

    assert plan.sections
    assert "比较对象" in plan.sections
    assert "考试总结" in plan.sections
    assert "结构" in plan.comparison_dimensions
    assert "长距离依赖" in plan.comparison_dimensions
    assert plan.retrieval_queries


def test_build_compare_prompt_contains_sections_and_dimensions() -> None:
    plan = build_compare_answer_plan("比较 accuracy precision recall F1", CompareTaskType.METRIC_COMPARISON)
    prompt = build_compare_prompt(
        "比较 accuracy precision recall F1",
        plan,
        retrieved_context="accuracy precision recall F1",
        evidence_status="partial",
        detail="standard",
    )

    for section in plan.sections:
        assert section in prompt
    for dimension in plan.comparison_dimensions:
        assert dimension in prompt
    assert "考试总结" in prompt
    assert "资料依据不足" in prompt
    assert "不允许伪造来源" in prompt


def test_solve_question_uses_compare_template(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_ask_question(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return make_ask_response(question)

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)
    response = solve_question(
        "比较 CNN 和 Transformer 的区别",
        mode="compare",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        detail="standard",
    )
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.COMPARE
    assert response.compare_task_type == CompareTaskType.MODEL_COMPARISON
    assert response.compare_sections is not None
    assert "考试总结" in response.compare_sections
    assert response.comparison_dimensions is not None
    assert "长距离依赖" in response.comparison_dimensions
    assert "比较维度" in str(seen["question"])
    assert "长距离依赖" in str(seen["question"])
    assert seen["detail"] == "standard"
    assert "对比任务类型：\nmodel_comparison" in rendered
    assert "比较维度：" in rendered
