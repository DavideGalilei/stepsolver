"""Deterministic fuzz regressions for exact answers that require human methods."""

import pytest

from stepsolver import ExactResult, Solver, format_latex_expression


@pytest.fixture
def solver() -> Solver:
    """Create the symbolic solver once per fuzz regression."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected_first_rule"),
    [
        ("sum(1/(2*n+1)^2,n,0,oo)", "Extract the odd terms of the p-series"),
        ("sum(1/(2*n)^2,n,1,oo)", "Extract the even terms of the p-series"),
        (
            "sum(1/((n+2)*(n+3)),n,0,oo)",
            "Decompose the summand into partial fractions",
        ),
        ("sum(n^2/2^n,n,1,oo)", "Differentiate the geometric-series identity twice"),
        (
            "sum((-1)^n/(2*n+1),n,0,oo)",
            "Apply the Gregory-Leibniz arctangent series",
        ),
        ("sum(1/factorial(2*n),n,0,oo)", "Apply the even exponential-series identity"),
        ("sum(1/factorial(2*n+1),n,0,oo)", "Apply the odd exponential-series identity"),
        ("diff(factorial(x),x)", "Rewrite the factorial with the gamma function"),
        ("diff(gamma(x),x)", "Differentiate gamma with the digamma function"),
        ("diff(abs(x),x)", "Differentiate the absolute value piecewise"),
        ("diff(tan(x),x)", "Differentiate the tangent"),
        ("diff(asin(x),x)", "Differentiate the inverse sine"),
        ("diff(acos(x),x)", "Differentiate the inverse cosine"),
        ("diff(atan(x),x)", "Differentiate the inverse tangent"),
        ("diff(sinh(x),x)", "Differentiate the hyperbolic sine"),
        ("diff(cosh(x),x)", "Differentiate the hyperbolic cosine"),
        ("diff(tanh(x),x)", "Differentiate the hyperbolic tangent"),
        ("diff(asinh(x),x)", "Differentiate the inverse hyperbolic sine"),
        ("integrate(x*log(x),x)", "Choose integration by parts"),
        ("integrate(x^2*log(x),x)", "Choose integration by parts"),
        ("integrate(tan(x),x)", "Use the logarithmic derivative of cosine"),
        (
            "integrate(1/sqrt(1-x^2),x)",
            "Use the inverse-sine derivative pattern",
        ),
        (
            "integrate(1/sqrt(1+x^2),x)",
            "Use the inverse-hyperbolic-sine derivative pattern",
        ),
        ("integrate(sinh(x),x)", "Use the hyperbolic sine antiderivative"),
        ("integrate(cosh(x),x)", "Use the hyperbolic cosine antiderivative"),
        (
            "integrate(x/sqrt(x^2+1),x)",
            "Use the reverse chain rule for the square root",
        ),
        (
            "limit(factorial(n+1)/factorial(n)/n,n,oo)",
            "Apply the factorial recurrence",
        ),
        ("limit(n!/((n+1)!),n,oo)", "Apply the factorial recurrence"),
        (
            "limit(n*(sqrt(n^2+1)-n),n,oo)",
            "Multiply by the conjugate",
        ),
    ],
)
def test_fuzz_discovered_exact_families_have_human_derivations(
    solver: Solver,
    query: str,
    expected_first_rule: str,
) -> None:
    """An exact result is a regression failure unless it has a recognizable method."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert result.steps
    assert result.steps[0].rule == expected_first_rule
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert all("symbolic backend" not in step.explanation.lower() for step in result.steps)
    assert all(step.before != step.after for step in result.steps)
    rendered = " ".join(
        format_latex_expression(expression)
        for step in result.steps
        for expression in (step.before, step.after)
    )
    assert r"\mathtt{" not in rendered


def test_absolute_value_derivative_states_its_nondifferentiable_point(
    solver: Solver,
) -> None:
    """The sign formula must not silently claim differentiability at the cusp."""
    result = solver.solve("diff(abs(x),x)")
    assert isinstance(result, ExactResult)
    assert result.steps[0].introduced_constraints
    assert (
        format_latex_expression(
            result.steps[0].introduced_constraints[0].expression,
        )
        == r"x \ne 0"
    )
