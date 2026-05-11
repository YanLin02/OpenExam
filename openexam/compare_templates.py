from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompareTaskType(str, Enum):
    MODEL_COMPARISON = "model_comparison"
    OPTIMIZER_COMPARISON = "optimizer_comparison"
    LOSS_COMPARISON = "loss_comparison"
    REGULARIZATION_COMPARISON = "regularization_comparison"
    METRIC_COMPARISON = "metric_comparison"
    GENERAL_COMPARISON = "general_comparison"


@dataclass(frozen=True)
class CompareAnswerPlan:
    task_type: CompareTaskType
    sections: list[str]
    guidance: str
    comparison_dimensions: list[str]
    retrieval_queries: list[str]


_SECTIONS = [
    "比较对象",
    "共同点",
    "主要区别",
    "优缺点",
    "适用场景",
    "考试总结",
]

_DIMENSIONS: dict[CompareTaskType, list[str]] = {
    CompareTaskType.MODEL_COMPARISON: [
        "结构",
        "输入假设",
        "特征建模方式",
        "参数共享",
        "长距离依赖",
        "计算复杂度",
        "适用任务",
    ],
    CompareTaskType.OPTIMIZER_COMPARISON: [
        "更新公式",
        "学习率敏感性",
        "收敛速度",
        "稳定性",
        "适用场景",
    ],
    CompareTaskType.LOSS_COMPARISON: [
        "数学形式",
        "适用任务",
        "对异常值敏感性",
        "概率解释",
        "梯度性质",
    ],
    CompareTaskType.REGULARIZATION_COMPARISON: [
        "作用机制",
        "对模型复杂度影响",
        "训练/推理差异",
        "使用成本",
        "适用场景",
    ],
    CompareTaskType.METRIC_COMPARISON: [
        "定义",
        "关注错误类型",
        "类别不平衡适用性",
        "解释方式",
        "使用场景",
    ],
    CompareTaskType.GENERAL_COMPARISON: [
        "定义",
        "原理",
        "优点",
        "缺点",
        "适用场景",
    ],
}

_GUIDANCE: dict[CompareTaskType, str] = {
    CompareTaskType.MODEL_COMPARISON: "按模型对比题作答，重点比较结构、建模方式、参数共享、长距离依赖、复杂度和适用任务。",
    CompareTaskType.OPTIMIZER_COMPARISON: "按优化器对比题作答，重点比较更新公式、学习率敏感性、收敛速度、稳定性和适用场景。",
    CompareTaskType.LOSS_COMPARISON: "按损失函数对比题作答，重点比较数学形式、适用任务、异常值敏感性、概率解释和梯度性质。",
    CompareTaskType.REGULARIZATION_COMPARISON: "按正则化方法对比题作答，重点比较作用机制、复杂度影响、训练推理差异、成本和适用场景。",
    CompareTaskType.METRIC_COMPARISON: "按评价指标对比题作答，重点比较定义、关注的错误类型、类别不平衡适用性、解释方式和使用场景。",
    CompareTaskType.GENERAL_COMPARISON: "按通用对比题作答，重点比较定义、原理、优缺点和适用场景。",
}


def classify_compare_task(question: str) -> CompareTaskType:
    text = question.strip().lower()
    if not text:
        return CompareTaskType.GENERAL_COMPARISON
    if any(keyword in text for keyword in ("accuracy", "precision", "recall", "f1", "评价指标")):
        return CompareTaskType.METRIC_COMPARISON
    if any(keyword in text for keyword in ("dropout", "l1", "l2", "正则化", "早停")):
        return CompareTaskType.REGULARIZATION_COMPARISON
    if any(keyword in text for keyword in ("mse", "交叉熵", "损失函数")):
        return CompareTaskType.LOSS_COMPARISON
    if any(keyword in text for keyword in ("sgd", "adam", "优化器", "动量")):
        return CompareTaskType.OPTIMIZER_COMPARISON
    if any(keyword in text for keyword in ("cnn", "rnn", "lstm", "transformer", "mlp", "卷积", "自注意力")):
        return CompareTaskType.MODEL_COMPARISON
    return CompareTaskType.GENERAL_COMPARISON


def build_compare_answer_plan(question: str, task_type: CompareTaskType) -> CompareAnswerPlan:
    sections = list(_SECTIONS)
    dimensions = list(_DIMENSIONS[task_type])
    retrieval_queries = [question, _GUIDANCE[task_type], " ".join(sections), " ".join(dimensions)]
    return CompareAnswerPlan(
        task_type=task_type,
        sections=sections,
        guidance=_GUIDANCE[task_type],
        comparison_dimensions=dimensions,
        retrieval_queries=retrieval_queries,
    )


def build_compare_prompt(
    question: str,
    plan: CompareAnswerPlan,
    retrieved_context: str,
    evidence_status: str,
    detail: str,
) -> str:
    sections = "\n".join(f"{index}. {section}" for index, section in enumerate(plan.sections, start=1))
    dimensions = "\n".join(f"- {dimension}" for dimension in plan.comparison_dimensions)
    detail_rules = {
        "concise": "简答版：每个部分 1-2 个核心要点。",
        "standard": "标准答案版：可用表格或分点，覆盖主要维度和考试总结。",
        "detailed": "复习展开版：按维度展开，但仍保持表格化或分点化。",
    }
    detail_rule = detail_rules.get(detail, detail_rules["standard"])
    context = retrieved_context.strip() or "由 OpenExam Ask 检索阶段提供；不得伪造未出现的来源。"
    status_note = "资料不足时也必须保持对比结构，并在第一行写“资料依据不足”。" if evidence_status != "sufficient" else "结论优先受本地资料约束。"
    return f"""原始对比题：
{question}

请按模板作答，尽量表格化或分点化，不要写成散文。
对比任务类型：{plan.task_type.value}
作答指导：{plan.guidance}
详细程度：{detail}；{detail_rule}
资料依据状态：{evidence_status}；{status_note}
本地检索提示：{context}

答题结构：
{sections}

比较维度：
{dimensions}

严格要求：
- 必须按上述“答题结构”顺序输出，且必须包含“考试总结”。
- “主要区别”部分要覆盖上方比较维度。
- 如果本地依据不足，必须显式标注“资料依据不足”。
- 不允许伪造来源、页码、文件名或引用编号。
- 不要输出隐藏思考过程。"""
