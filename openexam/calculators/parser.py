from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from openexam.calculators.cnn import conv2d_output_size, conv2d_param_count, linear_param_count, pool2d_output_size
from openexam.calculators.losses import cross_entropy_from_logits, mse, softmax
from openexam.calculators.metrics import classification_metrics
from openexam.calculators.optimization import gradient_descent_step
from openexam.calculators.sequence import lstm_param_count, simple_rnn_param_count
from openexam.calculators.transformer import attention_qkv_param_count, multihead_attention_param_count


@dataclass(frozen=True)
class CalculationParseResult:
    calculation_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    need_manual_input: bool = False
    reason: str = ""


@dataclass(frozen=True)
class CalculationAnswer:
    calculation_type: str
    formula: str
    substitution: str
    result_text: str
    values: dict[str, Any] = field(default_factory=dict)
    need_manual_input: bool = False
    reason: str = ""


NEED_MANUAL_INPUT = "need_manual_input"
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_PAIR = re.compile(r"(\d+)\s*[x×*]\s*(\d+)", re.IGNORECASE)


def _normalize(text: str) -> str:
    return text.strip().replace("，", ",").replace("：", ":").replace("；", ";")


def _first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _int_pair_after(label_pattern: str, text: str) -> tuple[int, int] | None:
    match = re.search(label_pattern + r"[^\d]{0,12}" + _PAIR.pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _all_pairs(text: str) -> list[tuple[int, int]]:
    return [(int(left), int(right)) for left, right in _PAIR.findall(text)]


def _optional_pair_value(pattern: str, text: str, default: int) -> tuple[int, int]:
    match = re.search(pattern + r"\s*=?\s*(\d+)(?:\s*[x×*]\s*(\d+))?", text, flags=re.IGNORECASE)
    if not match:
        return default, default
    first = int(match.group(1))
    second = int(match.group(2)) if match.group(2) is not None else first
    return first, second


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def detect_calculation_type(question: str) -> str:
    text = _normalize(question).lower()
    if not text:
        return NEED_MANUAL_INPUT
    if all(token.lower() in text for token in ("tp", "fp", "tn", "fn")):
        return "classification_metrics"
    if all(token in text for token in ("w", "grad", "lr")) or "梯度下降" in text:
        if re.search(r"\bw\s*=", text) and re.search(r"\bgrad\s*=", text) and re.search(r"\blr\s*=", text):
            return "gradient_descent_step"
    if "logits" in text and ("交叉熵" in text or "softmax" in text):
        return "softmax_cross_entropy"
    if "mse" in text or "均方误差" in text:
        return "mse"
    if ("全连接" in text or "linear" in text) and ("参数量" in text or "参数" in text):
        return "linear_param_count"
    if ("卷积" in text or "conv" in text) and "参数" in text and ("通道" in text or "channel" in text):
        return "conv2d_param_count"
    if ("池化" in text or "pool" in text) and ("输出尺寸" in text or "尺寸" in text):
        return "pool2d_output_size"
    if ("卷积" in text or "conv" in text) and ("输出尺寸" in text or "尺寸" in text):
        return "conv2d_output_size"
    if "lstm" in text and "参数" in text:
        return "lstm_param_count"
    if "rnn" in text and "参数" in text:
        return "simple_rnn_param_count"
    if ("qkv" in text or "q/k/v" in text or "q、k、v" in text) and "参数" in text:
        return "attention_qkv_param_count"
    if ("multihead" in text or "多头注意力" in text or "attention" in text) and "参数" in text:
        return "multihead_attention_param_count"
    return NEED_MANUAL_INPUT


def _parse_conv_or_pool_output(question: str, calculation_type: str) -> CalculationParseResult:
    input_pair = _int_pair_after(r"(?:输入|input)", question)
    kernel_pair = _int_pair_after(r"(?:卷积核|kernel|池化核|窗口)", question)
    if input_pair is None or kernel_pair is None:
        pairs = _all_pairs(question)
        if len(pairs) >= 2:
            input_pair = input_pair or pairs[0]
            kernel_pair = kernel_pair or pairs[1]
    if input_pair is None or kernel_pair is None:
        return CalculationParseResult(calculation_type, need_manual_input=True, reason="missing input or kernel size")
    if "无填充" in question:
        padding_h, padding_w = 0, 0
    else:
        padding_h, padding_w = _optional_pair_value(r"(?:padding|填充)", question, default=0)
    stride_h, stride_w = _optional_pair_value(r"(?:stride|步长)", question, default=1)
    dilation_h, dilation_w = _optional_pair_value(r"(?:dilation|膨胀)", question, default=1)
    return CalculationParseResult(
        calculation_type=calculation_type,
        parameters={
            "input_h": input_pair[0],
            "input_w": input_pair[1],
            "kernel_h": kernel_pair[0],
            "kernel_w": kernel_pair[1],
            "stride_h": stride_h,
            "stride_w": stride_w,
            "padding_h": padding_h,
            "padding_w": padding_w,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
        },
    )


def _parse_conv_params(question: str) -> CalculationParseResult:
    in_channels = _first_int(r"(?:输入通道|in_channels?|输入 channel)\s*=?\s*(\d+)", question)
    out_channels = _first_int(r"(?:输出通道|out_channels?|输出 channel)\s*=?\s*(\d+)", question)
    kernel_pair = _int_pair_after(r"(?:卷积核|kernel)", question)
    if in_channels is None or out_channels is None or kernel_pair is None:
        return CalculationParseResult("conv2d_param_count", need_manual_input=True, reason="missing channel or kernel size")
    groups = _first_int(r"(?:groups?|分组)\s*=?\s*(\d+)", question) or 1
    bias = not bool(re.search(r"(?:无|不含|没有)\s*bias|无偏置|不含偏置", question, flags=re.IGNORECASE))
    return CalculationParseResult(
        "conv2d_param_count",
        {"in_channels": in_channels, "out_channels": out_channels, "kernel_h": kernel_pair[0], "kernel_w": kernel_pair[1], "bias": bias, "groups": groups},
    )


def _parse_linear_params(question: str) -> CalculationParseResult:
    in_features = _first_int(r"(?:输入|in_features?)\s*=?\s*(\d+)", question)
    out_features = _first_int(r"(?:输出|out_features?)\s*=?\s*(\d+)", question)
    if in_features is None or out_features is None:
        return CalculationParseResult("linear_param_count", need_manual_input=True, reason="missing input or output features")
    bias = not bool(re.search(r"(?:无|不含|没有)\s*bias|无偏置|不含偏置", question, flags=re.IGNORECASE))
    return CalculationParseResult("linear_param_count", {"in_features": in_features, "out_features": out_features, "bias": bias})


def _parse_logits(question: str) -> CalculationParseResult:
    match = re.search(r"logits?\s*=\s*\[([^\]]+)\]", question, flags=re.IGNORECASE)
    target = _first_int(r"(?:真实类别|target|类别)\s*(?:为|=)?\s*(\d+)", question)
    if match is None or target is None:
        return CalculationParseResult("softmax_cross_entropy", need_manual_input=True, reason="missing logits or target index")
    values = [float(part) for part in re.findall(_NUMBER, match.group(1))]
    return CalculationParseResult("softmax_cross_entropy", {"logits": values, "target_index": target})


def _parse_metrics(question: str) -> CalculationParseResult:
    values = {}
    for name in ("tp", "fp", "tn", "fn"):
        value = _first_int(rf"\b{name}\s*=\s*(\d+)", question)
        if value is None:
            return CalculationParseResult("classification_metrics", need_manual_input=True, reason=f"missing {name.upper()}")
        values[name] = value
    return CalculationParseResult("classification_metrics", values)


def _parse_gradient_descent(question: str) -> CalculationParseResult:
    w = _first_float(r"\bw\s*=\s*(" + _NUMBER + ")", question)
    grad = _first_float(r"\bgrad\s*=\s*(" + _NUMBER + ")", question)
    lr = _first_float(r"\blr\s*=\s*(" + _NUMBER + ")", question)
    if w is None or grad is None or lr is None:
        return CalculationParseResult("gradient_descent_step", need_manual_input=True, reason="missing w, grad, or lr")
    return CalculationParseResult("gradient_descent_step", {"w": w, "grad": grad, "lr": lr})


def _float_list_after(label_pattern: str, text: str) -> list[float] | None:
    match = re.search(label_pattern + r"\s*=\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if not match:
        return None
    return [float(part) for part in re.findall(_NUMBER, match.group(1))]


def _parse_mse(question: str) -> CalculationParseResult:
    y_true = _float_list_after(r"(?:y_true|true|真实值)", question)
    y_pred = _float_list_after(r"(?:y_pred|pred|预测值)", question)
    if y_true is None or y_pred is None:
        return CalculationParseResult("mse", need_manual_input=True, reason="missing y_true or y_pred")
    return CalculationParseResult("mse", {"y_true": y_true, "y_pred": y_pred})


def _parse_rnn_params(question: str) -> CalculationParseResult:
    input_size = _first_int(r"(?:input_size|输入维度|输入大小)\s*=?\s*(\d+)", question)
    hidden_size = _first_int(r"(?:hidden_size|隐藏维度|隐状态维度|隐藏层大小)\s*=?\s*(\d+)", question)
    output_size = _first_int(r"(?:output_size|输出维度|输出大小)\s*=?\s*(\d+)", question)
    if input_size is None or hidden_size is None:
        return CalculationParseResult("simple_rnn_param_count", need_manual_input=True, reason="missing input_size or hidden_size")
    bias = not bool(re.search(r"(?:无|不含|没有)\s*bias|无偏置|不含偏置", question, flags=re.IGNORECASE))
    return CalculationParseResult(
        "simple_rnn_param_count",
        {"input_size": input_size, "hidden_size": hidden_size, "output_size": output_size, "bias": bias},
    )


def _parse_lstm_params(question: str) -> CalculationParseResult:
    input_size = _first_int(r"(?:input_size|输入维度|输入大小)\s*=?\s*(\d+)", question)
    hidden_size = _first_int(r"(?:hidden_size|隐藏维度|隐状态维度|隐藏层大小)\s*=?\s*(\d+)", question)
    if input_size is None or hidden_size is None:
        return CalculationParseResult("lstm_param_count", need_manual_input=True, reason="missing input_size or hidden_size")
    bias = not bool(re.search(r"(?:无|不含|没有)\s*bias|无偏置|不含偏置", question, flags=re.IGNORECASE))
    return CalculationParseResult("lstm_param_count", {"input_size": input_size, "hidden_size": hidden_size, "bias": bias})


def _parse_attention_params(question: str, calculation_type: str) -> CalculationParseResult:
    d_model = _first_int(r"(?:d_model|模型维度|隐藏维度)\s*=?\s*(\d+)", question)
    if d_model is None:
        return CalculationParseResult(calculation_type, need_manual_input=True, reason="missing d_model")
    bias = not bool(re.search(r"(?:无|不含|没有)\s*bias|无偏置|不含偏置", question, flags=re.IGNORECASE))
    if calculation_type == "attention_qkv_param_count":
        return CalculationParseResult(calculation_type, {"d_model": d_model, "bias": bias})
    include_output_projection = not bool(re.search(r"不含输出投影|不包括输出投影|without output projection", question, flags=re.IGNORECASE))
    return CalculationParseResult(
        calculation_type,
        {"d_model": d_model, "include_output_projection": include_output_projection, "bias": bias},
    )


def parse_calculation_question(question: str) -> CalculationParseResult:
    normalized = _normalize(question)
    calculation_type = detect_calculation_type(normalized)
    if calculation_type == "conv2d_output_size":
        return _parse_conv_or_pool_output(normalized, calculation_type)
    if calculation_type == "pool2d_output_size":
        return _parse_conv_or_pool_output(normalized, calculation_type)
    if calculation_type == "conv2d_param_count":
        return _parse_conv_params(normalized)
    if calculation_type == "linear_param_count":
        return _parse_linear_params(normalized)
    if calculation_type == "softmax_cross_entropy":
        return _parse_logits(normalized)
    if calculation_type == "classification_metrics":
        return _parse_metrics(normalized.lower())
    if calculation_type == "gradient_descent_step":
        return _parse_gradient_descent(normalized.lower())
    if calculation_type == "mse":
        return _parse_mse(normalized)
    if calculation_type == "simple_rnn_param_count":
        return _parse_rnn_params(normalized)
    if calculation_type == "lstm_param_count":
        return _parse_lstm_params(normalized)
    if calculation_type in {"attention_qkv_param_count", "multihead_attention_param_count"}:
        return _parse_attention_params(normalized, calculation_type)
    return CalculationParseResult(calculation_type, need_manual_input=True, reason="unsupported or ambiguous calculation question")


def _answer_from_dict(calculation_type: str, values: dict[str, object]) -> CalculationAnswer:
    return CalculationAnswer(
        calculation_type=calculation_type,
        formula=str(values["formula"]),
        substitution=str(values["substitution"]),
        result_text=str(values["result_text"]),
        values=values,
    )


def solve_calculation_question(question: str) -> CalculationAnswer:
    parsed = parse_calculation_question(question)
    if parsed.need_manual_input:
        return CalculationAnswer(
            calculation_type=parsed.calculation_type,
            formula="",
            substitution="",
            result_text="",
            need_manual_input=True,
            reason=parsed.reason,
        )
    params = parsed.parameters
    if parsed.calculation_type == "conv2d_output_size":
        return _answer_from_dict(parsed.calculation_type, conv2d_output_size(**params))
    if parsed.calculation_type == "pool2d_output_size":
        return _answer_from_dict(parsed.calculation_type, pool2d_output_size(**params))
    if parsed.calculation_type == "conv2d_param_count":
        return _answer_from_dict(parsed.calculation_type, conv2d_param_count(**params))
    if parsed.calculation_type == "linear_param_count":
        return _answer_from_dict(parsed.calculation_type, linear_param_count(**params))
    if parsed.calculation_type == "softmax_cross_entropy":
        logits = params["logits"]
        target_index = int(params["target_index"])
        probs = softmax(logits)
        loss = cross_entropy_from_logits(logits, target_index)
        formula = "softmax_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z_j)); CE = -log(p_target)"
        substitution = (
            f"logits = {logits}; softmax = {[round(prob, 6) for prob in probs]}; "
            f"CE = -log({_format_float(probs[target_index])}) = {_format_float(loss)}"
        )
        return CalculationAnswer(
            calculation_type=parsed.calculation_type,
            formula=formula,
            substitution=substitution,
            result_text=f"softmax = {[round(prob, 6) for prob in probs]}，交叉熵 = {_format_float(loss)}。",
            values={"probs": probs, "cross_entropy": loss, "logits": logits, "target_index": target_index},
        )
    if parsed.calculation_type == "classification_metrics":
        metrics = classification_metrics(**params)
        formula = "accuracy=(TP+TN)/(TP+FP+TN+FN); precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R)"
        substitution = (
            f"accuracy=({params['tp']}+{params['tn']})/({params['tp']}+{params['fp']}+{params['tn']}+{params['fn']})={_format_float(metrics['accuracy'])}\n"
            f"precision={params['tp']}/({params['tp']}+{params['fp']})={_format_float(metrics['precision'])}\n"
            f"recall={params['tp']}/({params['tp']}+{params['fn']})={_format_float(metrics['recall'])}\n"
            f"F1=2*P*R/(P+R)={_format_float(metrics['f1'])}"
        )
        return CalculationAnswer(
            calculation_type=parsed.calculation_type,
            formula=formula,
            substitution=substitution,
            result_text=(
                f"accuracy={_format_float(metrics['accuracy'])}, precision={_format_float(metrics['precision'])}, "
                f"recall={_format_float(metrics['recall'])}, F1={_format_float(metrics['f1'])}。"
            ),
            values=metrics,
        )
    if parsed.calculation_type == "gradient_descent_step":
        new_w = gradient_descent_step(**params)
        formula = "w_new = w - lr * grad"
        substitution = f"w_new = {params['w']} - {params['lr']} * {params['grad']} = {_format_float(new_w)}"
        return CalculationAnswer(
            calculation_type=parsed.calculation_type,
            formula=formula,
            substitution=substitution,
            result_text=f"梯度下降一步后的 w = {_format_float(new_w)}。",
            values={"w_new": new_w, **params},
        )
    if parsed.calculation_type == "mse":
        value = mse(**params)
        formula = "MSE = (1/n) * sum_i (y_true_i - y_pred_i)^2"
        substitution = f"MSE = mean(({params['y_true']} - {params['y_pred']})^2) = {_format_float(value)}"
        return CalculationAnswer(
            calculation_type=parsed.calculation_type,
            formula=formula,
            substitution=substitution,
            result_text=f"MSE = {_format_float(value)}。",
            values={"mse": value, **params},
        )
    if parsed.calculation_type == "simple_rnn_param_count":
        return _answer_from_dict(parsed.calculation_type, simple_rnn_param_count(**params))
    if parsed.calculation_type == "lstm_param_count":
        return _answer_from_dict(parsed.calculation_type, lstm_param_count(**params))
    if parsed.calculation_type == "attention_qkv_param_count":
        return _answer_from_dict(parsed.calculation_type, attention_qkv_param_count(**params))
    if parsed.calculation_type == "multihead_attention_param_count":
        return _answer_from_dict(parsed.calculation_type, multihead_attention_param_count(**params))
    return CalculationAnswer(
        calculation_type=parsed.calculation_type,
        formula="",
        substitution="",
        result_text="",
        need_manual_input=True,
        reason="calculation type has no dispatcher",
    )
