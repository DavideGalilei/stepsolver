"""ASCII and LaTeX precedence regressions for exponent grouping."""

import pytest

from stepsolver import (
    Solver,
    format_expression,
    format_latex_expression,
    parse_expression,
    solve_payload,
)


@pytest.mark.parametrize(
    ("source", "expected_ascii", "expected_latex"),
    [
        ("(2^3)^2", "(2 ^ 3) ^ 2", r"\left(2^{3}\right)^{2}"),
        ("2^(3^2)", "2 ^ 3 ^ 2", r"2^{3^{2}}"),
        ("(-3)^2", "(-3) ^ 2", r"\left(-3\right)^{2}"),
        ("(-x)^2", "(-x) ^ 2", r"\left(-x\right)^{2}"),
        ("-(x^2)", "-(x ^ 2)", r"-\left(x^{2}\right)"),
        ("2^(-3)", "2 ^ (-3)", r"2^{-3}"),
        ("x^(-y)", "x ^ (-y)", r"x^{-y}"),
    ],
)
def test_powers_keep_semantically_required_grouping(
    source: str,
    expected_ascii: str,
    expected_latex: str,
) -> None:
    """A rendered power must reparse to the same expression and keep its meaning."""
    expression = parse_expression(source)
    assert format_expression(expression) == expected_ascii
    assert format_latex_expression(expression) == expected_latex
    assert parse_expression(format_expression(expression)) == expression


@pytest.mark.parametrize(
    ("query", "expected_ascii", "expected_latex"),
    [
        ("simplify((2^3)^2)", "(2 ^ 3) ^ 2", r"\left(2^{3}\right)^{2}"),
        ("expand((-3)^2)", "(-3) ^ 2", r"\left(-3\right)^{2}"),
        ("simplify((-x)^2)", "(-x) ^ 2", r"\left(-x\right)^{2}"),
    ],
)
def test_frontend_payload_uses_grouped_power_bases(
    query: str,
    expected_ascii: str,
    expected_latex: str,
) -> None:
    """The public payload adapter must expose the corrected grouping, not only the helper."""
    payload = solve_payload(Solver().solve(query))
    assert payload.status == "exact"
    step = payload.steps[0]
    assert step.before_ascii == expected_ascii
    assert step.before_latex == expected_latex
