from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DesignTaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    SEQUENCE_MODELING = "sequence_modeling"
    GENERATIVE_MODEL = "generative_model"
    OVERFITTING_SOLUTION = "overfitting_solution"
    TRAINING_PIPELINE = "training_pipeline"
    GENERAL_NETWORK_DESIGN = "general_network_design"


@dataclass(frozen=True)
class DesignAnswerPlan:
    task_type: DesignTaskType
    sections: list[str]
    guidance: str
    retrieval_queries: list[str]


_SECTIONS: dict[DesignTaskType, list[str]] = {
    DesignTaskType.IMAGE_CLASSIFICATION: [
        "任务目标",
        "输入输出",
        "网络结构",
        "卷积层作用",
        "池化层作用",
        "全连接层与分类器",
        "损失函数",
        "优化方法",
        "防过拟合措施",
        "评价指标",
    ],
    DesignTaskType.SEQUENCE_MODELING: [
        "任务目标",
        "输入输出序列",
        "模型结构选择",
        "隐状态或注意力机制",
        "损失函数",
        "训练方式",
        "评价指标",
    ],
    DesignTaskType.GENERATIVE_MODEL: [
        "任务目标",
        "生成器/编码器结构",
        "判别器或解码器结构",
        "损失函数",
        "训练流程",
        "稳定训练策略",
        "评价方式",
    ],
    DesignTaskType.OVERFITTING_SOLUTION: [
        "过拟合现象",
        "原因分析",
        "数据增强",
        "L1/L2 正则化",
        "Dropout",
        "Early stopping",
        "验证集监控",
        "模型复杂度控制",
    ],
    DesignTaskType.TRAINING_PIPELINE: [
        "数据准备",
        "模型选择",
        "损失函数",
        "优化器",
        "训练循环",
        "验证与测试",
        "调参",
        "保存模型",
    ],
    DesignTaskType.GENERAL_NETWORK_DESIGN: [
        "任务目标",
        "输入输出",
        "模型结构",
        "关键模块",
        "损失函数",
        "优化方法",
        "评价指标",
        "改进方向",
    ],
}

_GUIDANCE: dict[DesignTaskType, str] = {
    DesignTaskType.IMAGE_CLASSIFICATION: "按图像分类网络设计题作答，重点覆盖 CNN 结构、特征提取、分类器、训练目标和评价指标。",
    DesignTaskType.SEQUENCE_MODELING: "按序列建模设计题作答，重点覆盖序列输入表示、RNN/LSTM/Transformer 选择、训练方式和评价指标。",
    DesignTaskType.GENERATIVE_MODEL: "按生成模型设计题作答，重点覆盖生成器/编码器、判别器/解码器、损失函数、训练流程和稳定性。",
    DesignTaskType.OVERFITTING_SOLUTION: "按防过拟合方案设计题作答，重点覆盖现象、原因、正则化、数据增强、Dropout、早停和验证集监控。",
    DesignTaskType.TRAINING_PIPELINE: "按训练流程设计题作答，重点覆盖数据、模型、损失、优化器、训练循环、验证测试和调参保存。",
    DesignTaskType.GENERAL_NETWORK_DESIGN: "按通用网络设计题作答，重点覆盖任务、输入输出、结构、关键模块、训练目标、评价和改进。",
}


def classify_design_task(question: str) -> DesignTaskType:
    text = question.strip().lower()
    if not text:
        return DesignTaskType.GENERAL_NETWORK_DESIGN
    if any(keyword in text for keyword in ("过拟合", "正则化", "dropout", "泛化", "early stopping")):
        return DesignTaskType.OVERFITTING_SOLUTION
    if any(keyword in text for keyword in ("gan", "生成", "vae", "扩散", "自编码器", "生成图像")):
        return DesignTaskType.GENERATIVE_MODEL
    if any(keyword in text for keyword in ("序列", "rnn", "lstm", "文本", "时间序列")) or ("transformer" in text and "文本" in text):
        return DesignTaskType.SEQUENCE_MODELING
    if any(keyword in text for keyword in ("训练流程", "实验流程", "优化器", "学习率", "验证集", "测试集")):
        return DesignTaskType.TRAINING_PIPELINE
    if any(keyword in text for keyword in ("图像分类", "cnn", "手写数字", "卷积", "图片", "图像")):
        return DesignTaskType.IMAGE_CLASSIFICATION
    return DesignTaskType.GENERAL_NETWORK_DESIGN


def build_design_answer_plan(question: str, task_type: DesignTaskType) -> DesignAnswerPlan:
    sections = list(_SECTIONS[task_type])
    retrieval_queries = [question, _GUIDANCE[task_type], " ".join(sections)]
    return DesignAnswerPlan(
        task_type=task_type,
        sections=sections,
        guidance=_GUIDANCE[task_type],
        retrieval_queries=retrieval_queries,
    )


def build_design_prompt(
    question: str,
    plan: DesignAnswerPlan,
    retrieved_context: str,
    evidence_status: str,
    detail: str,
) -> str:
    sections = "\n".join(f"{index}. {section}" for index, section in enumerate(plan.sections, start=1))
    detail_rules = {
        "concise": "考试简答版：每节 1-2 句，只保留可得分要点。",
        "standard": "标准答案版：每节 2-3 个要点，适合直接整理成考试答案。",
        "detailed": "复习展开版：每节可展开说明，但仍要结构化、可快速查阅。",
    }
    detail_rule = detail_rules.get(detail, detail_rules["standard"])
    context = retrieved_context.strip() or "由 OpenExam Ask 检索阶段提供；不得伪造未出现的来源。"
    status_note = "资料不足时也必须保持模板结构，并在第一行写“资料依据不足”。" if evidence_status != "sufficient" else "结论优先受本地资料约束。"
    return f"""原始设计题：
{question}

请按模板作答，必须分条，适合开卷考试快速使用。
设计任务类型：{plan.task_type.value}
作答指导：{plan.guidance}
详细程度：{detail}；{detail_rule}
资料依据状态：{evidence_status}；{status_note}
本地检索提示：{context}

答题结构：
{sections}

严格要求：
- 必须按上述“答题结构”顺序逐节输出，节标题必须一致。
- 如果本地依据不足，必须显式标注“资料依据不足”，但仍要逐节回答。
- 不允许伪造来源、页码、文件名或引用编号。
- 不要写成散文；不要输出思考过程。"""
