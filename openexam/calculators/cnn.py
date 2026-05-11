from __future__ import annotations

import math


def _output_dim(input_size: int, kernel_size: int, stride: int, padding: int, dilation: int) -> int:
    if min(input_size, kernel_size, stride, dilation) <= 0:
        raise ValueError("input, kernel, stride, and dilation must be positive integers.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")
    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def conv2d_output_size(
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int = 1,
    stride_w: int = 1,
    padding_h: int = 0,
    padding_w: int = 0,
    dilation_h: int = 1,
    dilation_w: int = 1,
) -> dict[str, object]:
    output_h = _output_dim(input_h, kernel_h, stride_h, padding_h, dilation_h)
    output_w = _output_dim(input_w, kernel_w, stride_w, padding_w, dilation_w)
    formula = "out = floor((in + 2p - d * (k - 1) - 1) / s) + 1"
    substitution = "\n".join(
        [
            f"H_out = floor(({input_h} + 2*{padding_h} - {dilation_h}*({kernel_h}-1) - 1) / {stride_h}) + 1 = {output_h}",
            f"W_out = floor(({input_w} + 2*{padding_w} - {dilation_w}*({kernel_w}-1) - 1) / {stride_w}) + 1 = {output_w}",
        ]
    )
    return {
        "output_h": output_h,
        "output_w": output_w,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"输出尺寸为 {output_h} x {output_w}。",
    }


def pool2d_output_size(
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int = 1,
    stride_w: int = 1,
    padding_h: int = 0,
    padding_w: int = 0,
    dilation_h: int = 1,
    dilation_w: int = 1,
) -> dict[str, object]:
    result = conv2d_output_size(
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
    )
    result["result_text"] = f"池化输出尺寸为 {result['output_h']} x {result['output_w']}。"
    return result


def conv2d_param_count(
    in_channels: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    bias: bool = True,
    groups: int = 1,
) -> dict[str, object]:
    if min(in_channels, out_channels, kernel_h, kernel_w, groups) <= 0:
        raise ValueError("channels, kernel sizes, and groups must be positive integers.")
    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups.")
    weight_params = out_channels * (in_channels // groups) * kernel_h * kernel_w
    bias_params = out_channels if bias else 0
    total = weight_params + bias_params
    formula = "params = out_channels * (in_channels / groups) * kernel_h * kernel_w + bias_params"
    substitution = (
        f"params = {out_channels} * ({in_channels} / {groups}) * {kernel_h} * {kernel_w}"
        f" + {bias_params} = {total}"
    )
    return {
        "params": total,
        "weight_params": weight_params,
        "bias_params": bias_params,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"卷积层参数量为 {total}。",
    }


def linear_param_count(in_features: int, out_features: int, bias: bool = True) -> dict[str, object]:
    if min(in_features, out_features) <= 0:
        raise ValueError("in_features and out_features must be positive integers.")
    weight_params = in_features * out_features
    bias_params = out_features if bias else 0
    total = weight_params + bias_params
    formula = "params = in_features * out_features + (out_features if bias else 0)"
    substitution = f"params = {in_features} * {out_features} + {bias_params} = {total}"
    return {
        "params": total,
        "weight_params": weight_params,
        "bias_params": bias_params,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"全连接层参数量为 {total}。",
    }
