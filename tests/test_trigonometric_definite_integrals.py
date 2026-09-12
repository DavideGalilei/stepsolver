"""Regressions for principal trigonometric square-root integrals."""

import pytest

from stepsolver import ExactResult, Solver, format_ascii, format_latex_expression


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("integrate(sqrt(1-cos(x)^2),x,0,2*pi)", "Result: 4"),
        ("integrate(sqrt(1-cos(t)^2),t,0,pi)", "Result: 2"),
        ("integrate(sqrt(1-cos(x)^2),x,pi,2*pi)", "Result: 2"),
        ("integrate(sqrt(1-cos(x)^2),x,0,pi/2)", "Result: 1"),
        ("integrate(sqrt(1-cos(x)^2),x,2*pi,4*pi)", "Result: 4"),
        ("integrate(abs(sin(x)),x,0,2*pi)", "Result: 4"),
    ],
)
def test_sine_square_root_family_uses_interval_signs(
    query: str,
    expected_result: str,
) -> None:
    """Pi-rational bounds and equivalent forms should receive exact human steps."""
    result = Solver().solve(query)

    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith(expected_result)
    assert result.steps
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert all(step.before != step.after for step in result.steps)


def test_reported_sine_square_root_integral_exposes_every_transformation() -> None:
    """The full-period case should retain the root, absolute value, split, and arithmetic."""
    result = Solver().solve("integrate(sqrt(1-cos(x)^2),x,0,2*pi)")

    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Use the Pythagorean identity",
        "Use the principal square root",
        "Split at the zeros of sine",
        "Remove the absolute value on each interval",
        "Evaluate each definite integral",
        "Add the interval contributions",
    )
    assert format_latex_expression(result.steps[0].after) == (
        r"\int_{0}^{2 \cdot \pi} \sqrt{\sin\left(x\right)^{2}}\,\mathrm{d}x"
    )
    assert format_latex_expression(result.steps[1].after) == (
        r"\int_{0}^{2 \cdot \pi} \left|\sin\left(x\right)\right|\,\mathrm{d}x"
    )
    assert format_latex_expression(result.steps[3].after) == (
        r"\int_{0}^{\pi} \sin\left(x\right)\,\mathrm{d}x + "
        r"\int_{\pi}^{2 \cdot \pi} -\sin\left(x\right)\,\mathrm{d}x"
    )
    assert format_latex_expression(result.steps[4].after) == "2 + 2"
    assert format_latex_expression(result.steps[5].after) == "4"
    assert result.steps[0].notes[0].label == "Pythagorean identity"
    assert result.steps[1].notes[0].label == "Principal-root rule"
    assert tuple(note.label for note in result.steps[3].notes) == (
        "Sign on interval 1",
        "Sign on interval 2",
    )
    assert any(not isinstance(part, str) for part in result.steps[0].explanation_parts)
    assert any(not isinstance(part, str) for part in result.steps[1].explanation_parts)
    assert "Abs(" not in format_ascii(result)
    assert "**" not in format_ascii(result)
