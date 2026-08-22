"""Regression coverage for explicit, student-facing L'Hopital derivations."""

import pytest

from stepsolver import ExactResult, Solver, solve_payload


@pytest.mark.parametrize(
    ("query", "expected_result", "rounds"),
    [
        ("limit((exp(x)-1)/x,x,0)", "1", 1),
        ("limit((exp(5*x)-1)/(2*x),x,0)", r"\frac{5}{2}", 1),
        ("limit(log(1+3*(x-2))/(x-2),x,2)", "3", 1),
        ("limit(sin(4*(x+1))/(3*(x+1)),x,-1)", r"\frac{4}{3}", 1),
        ("limit((1-cos(2*x))/x^2,x,0)", "2", 2),
    ],
)
def test_zero_over_zero_limits_show_the_complete_lhopital_method(
    query: str,
    expected_result: str,
    rounds: int,
) -> None:
    """A standard zero-over-zero limit must show why and how L'Hopital applies."""
    result = Solver().solve(query)
    assert isinstance(result, ExactResult)
    payload = solve_payload(result)

    expected_rules = ["Check direct substitution", "Apply L'Hôpital's rule"]
    expected_rules.extend("Apply L'Hôpital's rule again" for _ in range(rounds - 1))
    expected_rules.append("Substitute into the transformed limit")
    assert [step.rule for step in payload.steps] == expected_rules
    assert payload.result_latex == expected_result

    direct_substitution = payload.steps[0]
    assert direct_substitution.after_latex == r"\frac{0}{0}"
    assert [note.label for note in direct_substitution.notes] == [
        "Numerator",
        "Denominator",
    ]
    assert any(part.latex == r"\frac{0}{0}" for part in direct_substitution.explanation_parts)

    for derivative_step in payload.steps[1 : rounds + 1]:
        assert [note.label for note in derivative_step.notes] == [
            "Differentiate the numerator",
            "Differentiate the denominator",
        ]
        assert derivative_step.before_latex != derivative_step.after_latex

    final_step = payload.steps[-1]
    assert final_step.after_latex == expected_result
    assert [note.label for note in final_step.notes] == ["Standard identity check"]


def test_algebraic_zero_over_zero_keeps_the_simpler_factoring_method() -> None:
    """L'Hopital should not displace a clearer algebraic cancellation."""
    result = Solver().solve("limit((x^2-1)/(x-1),x,1)")
    assert isinstance(result, ExactResult)
    assert [step.rule for step in result.steps] == [
        "Factor and cancel the common factor",
        "Substitute into the simplified expression",
    ]
