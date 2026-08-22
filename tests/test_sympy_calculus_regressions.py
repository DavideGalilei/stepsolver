"""Human-method regressions adapted from the SymPy test suite.

The source project is Copyright 2006-2023 SymPy Development Team and uses
the BSD 3-Clause license. See THIRD_PARTY_NOTICES.md for pinned provenance.
Inputs use StepSolver syntax and assert StepSolver's independent derivations.
"""

import pytest

from stepsolver import ExactResult, Solver, format_ascii, format_expression


@pytest.fixture
def solver() -> Solver:
    """Create the default solver for the upstream-inspired corpus."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected_rules", "expected_result"),
    [
        (
            "integrate(log(x),x)",
            ("Choose integration by parts", "Evaluate the simpler remaining integral"),
            "Result: -x + x * log(x) + C",
        ),
        (
            "integrate(atan(x),x)",
            ("Choose integration by parts", "Evaluate the simpler remaining integral"),
            "Result: -log(x ^ 2 + 1) / 2 + x * atan(x) + C",
        ),
        (
            "integrate(exp(x)*sin(x),x)",
            (
                "Integrate by parts once",
                "Integrate by parts again and solve for the integral",
            ),
            "Result: exp(x) * sin(x) / 2 + -exp(x) * cos(x) / 2 + C",
        ),
        (
            "integrate(sin(2*x)*exp(x),x)",
            (
                "Integrate by parts once",
                "Integrate by parts again and solve for the integral",
            ),
            "Result: -2 * exp(x) * cos(2 * x) / 5 + exp(x) * sin(2 * x) / 5 + C",
        ),
        (
            "integrate(exp(x)*cos(x),x)",
            (
                "Integrate by parts once",
                "Integrate by parts again and solve for the integral",
            ),
            "Result: exp(x) * cos(x) / 2 + exp(x) * sin(x) / 2 + C",
        ),
        (
            "integrate(sin(x)^2*cos(x),x)",
            ("Substitute the sine", "Apply the power rule and substitute back"),
            "Result: sin(x) ^ 3 / 3 + C",
        ),
        (
            "integrate(sin(x)*cos(x)^3,x)",
            ("Substitute the cosine", "Apply the power rule and substitute back"),
            "Result: -(cos(x) ^ 4) / 4 + C",
        ),
        (
            "integrate(x/(x^4+1),x)",
            ("Substitute the repeated inner power", "Use the arctangent antiderivative"),
            "Result: atan(x ^ 2) / 2 + C",
        ),
    ],
)
def test_sympy_manual_integration_cases_show_the_classroom_method(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
    expected_result: str,
) -> None:
    """Manual-integration benchmarks should expose substitutions and parts choices."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert all(step.before != step.after for step in result.steps)
    assert format_ascii(result).endswith(expected_result)


def test_general_variable_power_uses_logarithmic_differentiation(solver: Solver) -> None:
    """A variable base and exponent require the general logarithmic power rule."""
    result = solver.solve("diff(x^x,x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Use logarithmic differentiation",)
    assert tuple(
        format_expression(constraint.expression)
        for constraint in result.steps[0].introduced_constraints
    ) == ("x > 0",)
    assert result.steps[0].notes[0].label == "General power rule"
    assert format_ascii(result).endswith("Result: x ^ x * (log(x) + 1)")


@pytest.mark.parametrize(
    ("query", "expected_rules", "expected_result"),
    [
        (
            "sum(1/(n*(n+1)),n,1,oo)",
            (
                "Decompose the summand into partial fractions",
                "Cancel the telescoping terms",
            ),
            "Result: 1",
        ),
        (
            "sum(1/(n*(n+1)),n,1,10)",
            (
                "Decompose the summand into partial fractions",
                "Cancel the telescoping terms",
            ),
            "Result: 10/11",
        ),
        (
            "sum(1/(n*(n+2)),n,1,oo)",
            (
                "Decompose the summand into partial fractions",
                "Cancel the telescoping terms",
            ),
            "Result: 3/4",
        ),
        (
            "sum(1/(n*(n+2)),n,1,10)",
            (
                "Decompose the summand into partial fractions",
                "Cancel the telescoping terms",
            ),
            "Result: 175/264",
        ),
        (
            "sum(n/2^n,n,1,oo)",
            ("Differentiate the geometric-series identity",),
            "Result: 2",
        ),
        (
            "sum(n*(1/2)^n,n,1,oo)",
            ("Differentiate the geometric-series identity",),
            "Result: 2",
        ),
    ],
)
def test_hard_series_use_telescoping_or_a_generating_identity(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
    expected_result: str,
) -> None:
    """Series benchmarks should state the cancellation or generating-function argument."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert format_ascii(result).endswith(expected_result)


@pytest.mark.parametrize(
    ("query", "expected_rules", "expected_result"),
    [
        (
            "limit((exp(x)-1)/x,x,0)",
            (
                "Check direct substitution",
                "Apply L'Hôpital's rule",
                "Substitute into the transformed limit",
            ),
            "Result: 1",
        ),
        (
            "limit(log(1+x)/x,x,0)",
            (
                "Check direct substitution",
                "Apply L'Hôpital's rule",
                "Substitute into the transformed limit",
            ),
            "Result: 1",
        ),
        (
            "limit((1-cos(x))/x^2,x,0)",
            (
                "Check direct substitution",
                "Apply L'Hôpital's rule",
                "Apply L'Hôpital's rule again",
                "Substitute into the transformed limit",
            ),
            "Result: 1/2",
        ),
        (
            "limit((1+1/x)^x,x,oo)",
            ("Use the exponential-definition limit",),
            "Result: E",
        ),
        (
            "limit(x^x,x,0,right)",
            ("Rewrite the variable power exponentially", "Evaluate the exponent limit"),
            "Result: 1",
        ),
        (
            "limit((sqrt(x+1)-1)/x,x,0)",
            ("Multiply by the conjugate", "Substitute into the rationalized expression"),
            "Result: 1/2",
        ),
    ],
)
def test_hard_limits_choose_the_shortest_standard_argument(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
    expected_result: str,
) -> None:
    """Indeterminate forms should use standard identities instead of backend evaluation."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert all(step.before != step.after for step in result.steps)
    assert format_ascii(result).endswith(expected_result)
