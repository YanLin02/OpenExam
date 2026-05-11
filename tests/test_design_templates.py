from __future__ import annotations

from openexam.ask import AskResponse
from openexam.config import AppConfig
from openexam.design_templates import (
    DesignTaskType,
    build_design_answer_plan,
    build_design_prompt,
    classify_design_task,
)
from openexam.problem_types import ProblemType
from openexam.solve import solve_question


def make_ask_response(question: str) -> AskResponse:
    return AskResponse(
        question=question,
        answer="1. 任务目标：完成分类。\n2. 输入输出：输入图像，输出类别。",
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


def test_classify_design_task_examples() -> None:
    assert classify_design_task("设计一个 CNN 完成手写数字识别任务") == DesignTaskType.IMAGE_CLASSIFICATION
    assert classify_design_task("设计一个 LSTM 文本分类模型") == DesignTaskType.SEQUENCE_MODELING
    assert classify_design_task("设计一个 GAN 生成图像") == DesignTaskType.GENERATIVE_MODEL
    assert classify_design_task("如何设计方案缓解过拟合") == DesignTaskType.OVERFITTING_SOLUTION
    assert classify_design_task("设计一个训练流程，包括优化器和验证集") == DesignTaskType.TRAINING_PIPELINE


def test_build_design_answer_plan_sections_are_not_empty() -> None:
    for task_type in DesignTaskType:
        plan = build_design_answer_plan("设计一个网络", task_type)
        assert plan.task_type == task_type
        assert plan.sections
        assert plan.guidance
        assert plan.retrieval_queries


def test_image_classification_plan_contains_key_sections() -> None:
    plan = build_design_answer_plan("设计一个 CNN 完成手写数字识别任务", DesignTaskType.IMAGE_CLASSIFICATION)

    assert "网络结构" in plan.sections
    assert "损失函数" in plan.sections
    assert "评价指标" in plan.sections


def test_overfitting_plan_contains_regularization_sections() -> None:
    plan = build_design_answer_plan("如何设计方案缓解过拟合", DesignTaskType.OVERFITTING_SOLUTION)

    assert "Dropout" in plan.sections
    assert "L1/L2 正则化" in plan.sections
    assert "Early stopping" in plan.sections


def test_build_design_prompt_contains_all_sections() -> None:
    plan = build_design_answer_plan("设计一个 CNN 完成手写数字识别任务", DesignTaskType.IMAGE_CLASSIFICATION)
    prompt = build_design_prompt(
        "设计一个 CNN 完成手写数字识别任务",
        plan,
        retrieved_context="CNN 卷积 池化 分类器",
        evidence_status="partial",
        detail="concise",
    )

    for section in plan.sections:
        assert section in prompt
    assert "资料依据不足" in prompt
    assert "考试简答版" in prompt
    assert "不允许伪造来源" in prompt


def test_solve_question_uses_design_template(monkeypatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    def fake_ask_question(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return make_ask_response(question)

    monkeypatch.setattr("openexam.solve.ask_question", fake_ask_question)

    response = solve_question(
        "设计一个 CNN 完成手写数字识别任务",
        mode="design",
        config=AppConfig(index_dir=tmp_path / ".openexam"),
        detail="concise",
    )

    assert response.problem_type == ProblemType.DESIGN
    assert response.design_task_type == DesignTaskType.IMAGE_CLASSIFICATION
    assert response.design_sections is not None
    assert "网络结构" in response.design_sections
    assert "答题结构" in str(seen["question"])
    assert "网络结构" in str(seen["question"])
    assert seen["detail"] == "concise"
