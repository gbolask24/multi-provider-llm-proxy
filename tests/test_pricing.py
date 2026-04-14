from app.providers.base import Usage


def test_estimate_cost_known_model():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=1000, output_tokens=500, total_tokens=1500)
    cost = estimate_cost("gpt-4o-mini", usage)
    # (1000 * 0.00000015) + (500 * 0.0000006) = 0.00015 + 0.0003 = 0.00045
    assert cost is not None
    assert abs(cost - 0.00045) < 1e-10


def test_estimate_cost_unknown_model_returns_none():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    cost = estimate_cost("unknown-model-xyz", usage)
    assert cost is None


def test_estimate_cost_missing_input_tokens():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=None, output_tokens=50, total_tokens=50)
    cost = estimate_cost("gpt-4o-mini", usage)
    assert cost is None


def test_estimate_cost_missing_output_tokens():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=100, output_tokens=None, total_tokens=100)
    cost = estimate_cost("gpt-4o-mini", usage)
    assert cost is None


def test_estimate_cost_zero_tokens():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    cost = estimate_cost("gpt-4o-mini", usage)
    assert cost == 0.0


def test_estimate_cost_anthropic_model():
    from app.pricing import estimate_cost

    usage = Usage(input_tokens=200, output_tokens=100, total_tokens=300)
    cost = estimate_cost("claude-3-haiku-20240307", usage)
    # (200 * 0.00000025) + (100 * 0.00000125) = 0.00005 + 0.000125 = 0.000175
    assert cost is not None
    assert abs(cost - 0.000175) < 1e-10
