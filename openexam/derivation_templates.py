from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DerivationTaskType(str, Enum):
    BACKPROPAGATION = "backpropagation"
    SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"
    GRADIENT_DESCENT = "gradient_descent"
    CNN_DIMENSION = "cnn_dimension"
    GENERAL_FORMULA = "general_formula"


@dataclass(frozen=True)
class DerivationAnswerPlan:
    task_type: DerivationTaskType
    sections: list[str]
    guidance: str
    retrieval_queries: list[str]


_SECTIONS: dict[DerivationTaskType, list[str]] = {
    DerivationTaskType.BACKPROPAGATION: [
        "推导目标",
        "变量与符号定义",
        "前向计算关系",
        "损失函数",
        "链式法则",
        "梯度表达式",
        "参数更新",
        "结论",
    ],
    DerivationTaskType.SOFTMAX_CROSS_ENTROPY: [
        "推导目标",
        "Softmax 定义",
        "交叉熵损失",
        "对 logits 求导",
        "化简结果",
        "结论",
    ],
    DerivationTaskType.GRADIENT_DESCENT: [
        "优化目标",
        "梯度定义",
        "更新公式",
        "学习率作用",
        "收敛直观解释",
        "结论",
    ],
    DerivationTaskType.CNN_DIMENSION: [
        "推导目标",
        "输入与卷积核参数",
        "padding / stride / dilation 作用",
        "输出尺寸公式",
        "特殊情况说明",
        "结论",
    ],
    DerivationTaskType.GENERAL_FORMULA: [
        "推导目标",
        "已知条件",
        "关键公式",
        "推导步骤",
        "最终结论",
        "适用条件",
    ],
}

_GUIDANCE: dict[DerivationTaskType, str] = {
    DerivationTaskType.BACKPROPAGATION: "按反向传播推导题作答，先定义符号和前向关系，再用链式法则逐步给出梯度与参数更新。",
    DerivationTaskType.SOFTMAX_CROSS_ENTROPY: "按 Softmax 与交叉熵梯度推导题作答，重点写清定义、求导、化简和最终梯度形式。",
    DerivationTaskType.GRADIENT_DESCENT: "按梯度下降推导题作答，重点说明优化目标、梯度方向、学习率和更新公式。",
    DerivationTaskType.CNN_DIMENSION: "按卷积输出尺寸公式推导题作答，重点解释 padding、stride、dilation 如何影响有效输入范围和输出个数。",
    DerivationTaskType.GENERAL_FORMULA: "按通用公式推导题作答，重点列明已知条件、关键公式、可见推导步骤、结论和适用条件。",
}


def classify_derivation_task(question: str) -> DerivationTaskType:
    text = question.strip().lower()
    if not text:
        return DerivationTaskType.GENERAL_FORMULA
    if any(keyword in text for keyword in ("反向传播", "bp", "链式法则")):
        return DerivationTaskType.BACKPROPAGATION
    if any(keyword in text for keyword in ("梯度下降", "参数更新", "优化")):
        return DerivationTaskType.GRADIENT_DESCENT
    if any(keyword in text for keyword in ("卷积", "输出尺寸", "尺寸公式")):
        return DerivationTaskType.CNN_DIMENSION
    if "softmax" in text or "交叉熵" in text or "梯度" in text:
        return DerivationTaskType.SOFTMAX_CROSS_ENTROPY
    return DerivationTaskType.GENERAL_FORMULA


def build_derivation_answer_plan(question: str, task_type: DerivationTaskType) -> DerivationAnswerPlan:
    sections = list(_SECTIONS[task_type])
    retrieval_queries = [question, _GUIDANCE[task_type], " ".join(sections)]
    return DerivationAnswerPlan(
        task_type=task_type,
        sections=sections,
        guidance=_GUIDANCE[task_type],
        retrieval_queries=retrieval_queries,
    )


def build_derivation_prompt(
    question: str,
    plan: DerivationAnswerPlan,
    retrieved_context: str,
    evidence_status: str,
    detail: str,
) -> str:
    sections = "\n".join(f"{index}. {section}" for index, section in enumerate(plan.sections, start=1))
    detail_rules = {
        "concise": "只给主公式和关键步骤，避免长篇解释。",
        "standard": "给完整考试答案，包含必要符号定义、推导步骤和结论。",
        "detailed": "给更细解释，但仍保持结构化，不展开隐藏思考过程。",
    }
    detail_rule = detail_rules.get(detail, detail_rules["standard"])
    context = retrieved_context.strip() or "由 OpenExam Ask 检索阶段提供；不得伪造未出现的来源。"
    status_note = "资料不足时也必须保持推导结构，并在第一行写“资料依据不足”。" if evidence_status != "sufficient" else "结论优先受本地资料约束。"
    return f"""原始推导题：
{question}

请按模板作答，必须输出可见推导步骤，不要输出隐藏思考过程。
推导任务类型：{plan.task_type.value}
作答指导：{plan.guidance}
详细程度：{detail}；{detail_rule}
资料依据状态：{evidence_status}；{status_note}
本地检索提示：{context}

推导结构：
{sections}

严格要求：
- 必须按上述“推导结构”顺序逐节输出，节标题必须一致。
- 必须区分“已知条件 / 推导步骤 / 结论”，即使模板标题不同也要在内容中体现。
- 如果本地依据不足，必须显式标注“资料依据不足”。
- 不允许伪造来源、页码、文件名或引用编号。
- 不要写成散文；不要输出隐藏思考过程。"""
