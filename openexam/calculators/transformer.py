from __future__ import annotations


def attention_qkv_param_count(d_model: int, bias: bool = True) -> dict[str, object]:
    if d_model <= 0:
        raise ValueError("d_model must be a positive integer.")
    bias_terms = d_model if bias else 0
    total = 3 * (d_model * d_model + bias_terms)
    formula = "params = 3 * (d_model * d_model + bias_terms)"
    substitution = f"params = 3 * ({d_model}*{d_model} + {bias_terms}) = {total}"
    return {
        "params": total,
        "bias_terms": 3 * bias_terms,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"Q/K/V 三个线性层参数量为 {total}。",
    }


def multihead_attention_param_count(
    d_model: int,
    include_output_projection: bool = True,
    bias: bool = True,
) -> dict[str, object]:
    qkv = attention_qkv_param_count(d_model, bias=bias)
    output_projection = d_model * d_model + (d_model if bias else 0) if include_output_projection else 0
    total = int(qkv["params"]) + output_projection
    formula = "params = qkv_params + output_projection_params"
    substitution = f"params = {qkv['params']} + {output_projection} = {total}"
    return {
        "params": total,
        "qkv_params": qkv["params"],
        "output_projection_params": output_projection,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"多头注意力参数量为 {total}。",
    }
