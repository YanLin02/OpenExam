from __future__ import annotations


def simple_rnn_param_count(input_size: int, hidden_size: int, output_size: int | None = None, bias: bool = True) -> dict[str, object]:
    if min(input_size, hidden_size) <= 0:
        raise ValueError("input_size and hidden_size must be positive integers.")
    input_to_hidden = input_size * hidden_size
    hidden_to_hidden = hidden_size * hidden_size
    bias_params = hidden_size if bias else 0
    output_params = 0
    output_bias = 0
    if output_size is not None:
        if output_size <= 0:
            raise ValueError("output_size must be a positive integer.")
        output_params = hidden_size * output_size
        output_bias = output_size if bias else 0
    total = input_to_hidden + hidden_to_hidden + bias_params + output_params + output_bias
    formula = "params = input_size*hidden_size + hidden_size*hidden_size + bias + optional_output"
    substitution = (
        f"params = {input_size}*{hidden_size} + {hidden_size}*{hidden_size} + {bias_params}"
        f" + {output_params} + {output_bias} = {total}"
    )
    return {
        "params": total,
        "input_to_hidden": input_to_hidden,
        "hidden_to_hidden": hidden_to_hidden,
        "bias_params": bias_params + output_bias,
        "output_params": output_params,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"简单 RNN 参数量为 {total}。",
    }


def lstm_param_count(input_size: int, hidden_size: int, bias: bool = True) -> dict[str, object]:
    if min(input_size, hidden_size) <= 0:
        raise ValueError("input_size and hidden_size must be positive integers.")
    bias_terms = hidden_size if bias else 0
    total = 4 * (input_size * hidden_size + hidden_size * hidden_size + bias_terms)
    formula = "params = 4 * (input_size * hidden_size + hidden_size * hidden_size + bias_terms)"
    substitution = f"params = 4 * ({input_size}*{hidden_size} + {hidden_size}*{hidden_size} + {bias_terms}) = {total}"
    return {
        "params": total,
        "bias_terms": bias_terms,
        "formula": formula,
        "substitution": substitution,
        "result_text": f"LSTM 参数量为 {total}。",
    }
