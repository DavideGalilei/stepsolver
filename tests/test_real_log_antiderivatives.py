"""Real-domain logarithmic antiderivatives for rational integrands."""

import json

import pytest
import sympy as sp

from stepsolver import ExactResult, Solver, format_ascii, solve_payload
from stepsolver.formatter import format_expression
from stepsolver.sympy_integrals import match_partial_fraction_antiderivative
from stepsolver.sympy_support import differentiates_on_real_intervals


@pytest.fixture
def solver() -> Solver:
    """Use the public solver configured with the default SymPy backend."""
    return Solver()


def _constraints(result: ExactResult) -> tuple[str, ...]:
    return tuple(
        format_expression(item.expression)
        for step in result.steps
        for item in step.introduced_constraints
    )


def test_r06_uses_absolute_logarithms_and_reports_both_poles(solver: Solver) -> None:
    """R06 must stay real across both poles and state both exclusions."""
    result = solver.solve("integrate(1/(x^2 - 1), x)")

    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Decompose into partial fractions",
        "Integrate each partial fraction",
    )
    assert _constraints(result) == ("x != -1", "x != 1")
    rendered = format_ascii(result)
    assert "log(abs(x - 1))" in rendered
    assert "log(abs(x + 1))" in rendered
    assert "Compute exact result" not in rendered


@pytest.mark.parametrize(
    ("query", "expected_constraints", "expected_logs"),
    [
        (
            "integrate(1/(x^2 - 4), x)",
            ("x != -2", "x != 2"),
            ("log(abs(x - 2))", "log(abs(x + 2))"),
        ),
        (
            "integrate(1/(x^2 + x), x)",
            ("x != -1", "x != 0"),
            ("log(abs(x))", "log(abs(x + 1))"),
        ),
        (
            "integrate(1/(x*(x + 1)), x)",
            ("x != -1", "x != 0"),
            ("log(abs(x))", "log(abs(x + 1))"),
        ),
        (
            "integrate(3/(x^2 - 9), x)",
            ("x != -3", "x != 3"),
            ("log(abs(x - 3))", "log(abs(x + 3))"),
        ),
        (
            "integrate((x + 2)/(x^2 - 1), x)",
            ("x != -1", "x != 1"),
            ("log(abs(x - 1))", "log(abs(x + 1))"),
        ),
    ],
)
def test_nearby_rational_integrals_use_real_logarithms(
    solver: Solver,
    query: str,
    expected_constraints: tuple[str, ...],
    expected_logs: tuple[str, ...],
) -> None:
    """The real-log rule must generalise beyond the reported denominator."""
    result = solver.solve(query)

    assert isinstance(result, ExactResult)
    assert _constraints(result) == expected_constraints
    rendered = format_ascii(result)
    for logarithm in expected_logs:
        assert logarithm in rendered
    assert result.steps[-1].verification.detail


def test_partial_fraction_matcher_builds_monic_absolute_logarithms() -> None:
    """The matcher returns one integrable term per factor with a real logarithm."""
    variable = sp.Symbol("x", real=True)
    matched = match_partial_fraction_antiderivative(1 / (variable**2 - 1), variable)

    assert matched is not None
    assert matched.result.has(sp.Abs)
    assert all(not term.has(sp.log) for term in matched.terms)
    rendered_terms = tuple(str(item) for item in matched.term_antiderivatives)
    assert "log(Abs(x + 1))" in " ".join(rendered_terms)
    assert "log(Abs(x - 1))" in " ".join(rendered_terms)


def test_interval_verification_rejects_an_inconsistent_sign_combination() -> None:
    """A candidate that is only correct on some intervals must be rejected."""
    variable = sp.Symbol("x", real=True)
    integrand = 1 / (variable**2 - 1)
    correct = sp.log(sp.Abs(variable - 1)) / 2 - sp.log(sp.Abs(variable + 1)) / 2
    inconsistent = sp.log(sp.Abs(variable - 1)) / 2 + sp.log(sp.Abs(variable + 1)) / 2

    assert differentiates_on_real_intervals(correct, integrand, variable)
    assert not differentiates_on_real_intervals(inconsistent, integrand, variable)


def test_r06_payload_serialises_real_logarithms_and_constraints(solver: Solver) -> None:
    """The browser payload must preserve the absolute values and exclusions."""
    result = solver.solve("integrate(1/(x^2 - 1), x)")
    assert isinstance(result, ExactResult)

    payload = solve_payload(result)
    encoded = json.loads(json.dumps(payload.as_dict()))

    assert payload.status == "exact"
    assert payload.result_latex is not None
    assert "\\left|x - 1\\right|" in payload.result_latex
    assert "\\left|x + 1\\right|" in payload.result_latex
    assert encoded["status"] == "exact"
    assert encoded["steps"][-1]["introduced_constraints"] == [
        {
            "explanation": "The original denominator cannot equal zero.",
            "expression_ascii": "x != -1",
            "expression_latex": "x \\ne -1",
        },
        {
            "explanation": "The original denominator cannot equal zero.",
            "expression_ascii": "x != 1",
            "expression_latex": "x \\ne 1",
        },
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("integrate(1/x, x)", "log(abs(x)) + C"),
        ("integrate(2/(2*x + 1), x)", "log(abs(2 * x + 1)) + C"),
    ],
)
def test_g04_and_g08_keep_their_absolute_logarithms(
    solver: Solver,
    query: str,
    expected: str,
) -> None:
    """The new rational-log path must not disturb the reciprocal-linear fast path."""
    result = solver.solve(query)

    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith(f"Result: {expected}")
