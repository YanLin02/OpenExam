from __future__ import annotations

from openexam.ask import AskResponse
from openexam.config import AppConfig
from openexam.derivation_templates import (
    DerivationTaskType,
    build_derivation_answer_plan,
    build_derivation_prompt,
    classify_derivation_task,
)
from openexam.problem_types import ProblemType
from openexam.solve import render_solve_response, solve_question


def make_ask_response(question: str) -> AskResponse:
    return AskResponse(
        question=question,
        answer="1. 推导目标：求梯度。\n2. 结论：得到可用于考试作答的公式。",
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


def test_classify_derivation_task_examples() -> None:
    assert classify_derivation_task("推导 softmax 交叉熵的梯度") == DerivationTaskType.SOFTMAX_CROSS_ENTROPY
    assert classify_derivation_task("推导误差反向传播算法") == DerivationTaskType.BACKPROPAGATION
    assert classify_derivation_task("推导梯度下降更新公式") == DerivationTaskType.GRADIENT_DESCENT
    assert classify_derivation_task("推导卷积输出尺寸公式") == DerivationTaskType.CNN_DIMENSION


def test_build_derivation_answer_plan_contains_key_sections() -> None:
    plan = build_derivation_answer_plan("推导 softmax 交叉熵的梯度", DerivationTaskType.SOFTMAX_CROSS_ENTROPY)

    assert plan.sections
    assert "推导目标" in plan.sections
    assert "对 logits 求导" in plan.sections
    assert "结论" in plan.sections
    assert plan.retrieval_queries


def test_build_derivation_prompt_contains_all_sections() -> None:
    plan = build_derivation_answer_plan("推导误差反向传播算法", DerivationTaskType.BACKPROPAGATION)
    prompt = build_derivation_prompt(
        "推导误差反向传播算法",
        plan,
        retrieved_context="反向传播 链式法则 梯度",
        evidence_status="partial",
        detail="standard",
    )

    for section in plan.sections:
        assert section in prompt
    assert "资料依据不足" in prompt
    assert "已知条件 / 推导步骤 / 结论" in prompt
    assert "不要输出隐藏思考过程" in prompt


def test_solve_question_uses_derivation_template(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_ask_question(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return make_ask_response(question)

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)
    response = solve_question(
        "推导 softmax 交叉熵的梯度",
        mode="derivation",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        detail="standard",
    )
    rendered = render_solve_response(response)

    assert response.problem_type == ProblemType.DERIVATION
    assert response.derivation_task_type == DerivationTaskType.SOFTMAX_CROSS_ENTROPY
    assert response.derivation_sections is not None
    assert "对 logits 求导" in response.derivation_sections
    assert "推导结构" in str(seen["question"])
    assert "Softmax 定义" in str(seen["question"])
    assert seen["detail"] == "standard"
    assert "推导任务类型：\nsoftmax_cross_entropy" in rendered
    assert "推导结构：" in rendered
