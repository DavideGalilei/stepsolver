"""Prompt 7 regressions: keep verified results and broaden inequality/integral steps."""

import pytest

from stepsolver import (
    DivergentResult,
    ExactResult,
    Solver,
    format_ascii,
    format_expression,
)

_OPAQUE_RULE = "Compute exact result"


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected", "boundary_rule"),
    [
        (
            "solve_inequality((x^2 - 1)/(x + 2) > 0, x)",
            "(1 < x) | ((-2 < x) & (x < -1))",
            (
                "A rational expression can change sign only where its numerator is zero "
                "or its denominator is zero."
            ),
        ),
        (
            "solve_inequality(abs(x - 1) < 2, x)",
            "(-1 < x) & (x < 3)",
            (
                "An absolute-value expression can change sign only where the whole expression "
                "equals zero"
            ),
        ),
        (
            "solve_inequality(abs(x^2 - 4) >= 1, x)",
            "(sqrt(5) <= x) | (x <= -sqrt(5)) | ((x <= sqrt(3)) & (-sqrt(3) <= x))",
            (
                "An absolute-value expression can change sign only where the whole expression "
                "equals zero"
            ),
        ),
    ],
)
def test_rational_and_absolute_value_inequalities_use_a_sign_chart(
    solver: Solver,
    query: str,
    expected: str,
    boundary_rule: str,
) -> None:
    """These inequalities should expose boundaries, test points, and an exact set."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert expected in format_ascii(result)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Move every term to one side",
        "Find the boundary points",
        "Test one point in each interval",
    )
    assert boundary_rule in result.steps[1].explanation
    assert result.steps[2].notes
    assert all(step.rule != _OPAQUE_RULE for step in result.steps)


def test_rational_inequality_excludes_its_pole(solver: Solver) -> None:
    """The excluded denominator zero must be labelled as a pole, not a boundary root."""
    result = solver.solve("solve_inequality((x^2 - 1)/(x + 2) > 0, x)")
    assert isinstance(result, ExactResult)
    labels = tuple(note.label for note in result.steps[1].notes)
    assert "Undefined pole at -2" in labels
    assert "Zero at -1" in labels
    assert "Zero at 1" in labels


@pytest.mark.parametrize(
    ("query", "expected_rule", "expected_text"),
    [
        (
            "solve_inequality(x^2 + 1 > 0, x)",
            "Check the sign for every real number",
            "all real numbers",
        ),
        (
            "solve_inequality(x^2 + 1 < 0, x)",
            "Check the sign for every real number",
            "no real solutions",
        ),
        (
            "solve_inequality(abs(x) < -1, x)",
            "Check the sign for every real number",
            "no real solutions",
        ),
    ],
)
def test_constant_sign_inequalities_explain_the_result(
    solver: Solver,
    query: str,
    expected_rule: str,
    expected_text: str,
) -> None:
    """An inequality with no sign change should still show why it holds or fails."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Move every term to one side",
        expected_rule,
    )
    assert expected_text in format_ascii(result)
    assert result.steps[-1].notes


def test_rational_log_antiderivative_is_kept_with_real_poles(solver: Solver) -> None:
    """A verified multi-pole logarithm is retained with every real exclusion."""
    result = solver.solve("integrate(1/(x^2 - 1), x)")
    assert isinstance(result, ExactResult)
    assert result.steps
    constraints = result.steps[0].introduced_constraints
    assert tuple(format_expression(item.expression) for item in constraints) == (
        "x != -1",
        "x != 1",
    )
    assert result.steps[0].rule == "Verify the antiderivative by differentiation"
    assert _OPAQUE_RULE not in format_ascii(result)


def test_integral_without_a_template_keeps_a_verified_answer(solver: Solver) -> None:
    """An exact antiderivative should survive the absence of a specific template."""
    result = solver.solve("integrate(1/(x^4 + 1), x)")
    assert isinstance(result, ExactResult)
    assert result.steps
    assert result.steps[0].rule == "Verify the antiderivative by differentiation"
    assert result.steps[0].verification.detail
    assert _OPAQUE_RULE not in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "expected_rules"),
    [
        (
            "integrate(asin(x), x)",
            ("Choose integration by parts", "Evaluate the simpler remaining integral"),
        ),
        (
            "integrate(acos(x), x)",
            ("Choose integration by parts", "Evaluate the simpler remaining integral"),
        ),
        (
            "integrate(x^2*atan(x), x)",
            ("Choose integration by parts", "Evaluate the remaining integral"),
        ),
        (
            "integrate(x*asin(x), x)",
            ("Choose integration by parts", "Evaluate the remaining integral"),
        ),
    ],
)
def test_inverse_trigonometric_integrals_use_parts(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
) -> None:
    """Inverse trigonometric integrands should expose a genuine by-parts choice."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert any(note.label == "Integration by parts" for note in result.steps[0].notes)
    assert any(
        note.label in {"Choose u", "Choose the inverse function"} for note in result.steps[0].notes
    )


def test_scaled_inverse_sine_uses_the_matching_derivative_pair(solver: Solver) -> None:
    """A scaled semicircle denominator should not fall back to a generic answer."""
    result = solver.solve("integrate(3/sqrt(9 - 4*x^2), x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Use the scaled inverse-sine antiderivative",
    )
    assert "asin(2 * x / 3)" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "expected_rules"),
    [
        (
            "integrate(1/sqrt(x), x, 0, 1)",
            (
                "Rewrite the improper integral as a limit",
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the limit",
            ),
        ),
        (
            "integrate(log(x), x, 0, 1)",
            (
                "Rewrite the improper integral as a limit",
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the limit",
            ),
        ),
    ],
)
def test_finite_endpoint_singularities_use_a_one_sided_limit(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
) -> None:
    """An integrand unbounded at a finite endpoint should be evaluated by a limit."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert result.steps[0].verification.detail
    assert result.steps[-1].after


def test_divergent_finite_endpoint_singularity_reports_the_infinity(
    solver: Solver,
) -> None:
    """A logarithmic singularity at zero should keep its divergent limit working."""
    result = solver.solve("integrate(1/x, x, 0, 1)")
    assert isinstance(result, DivergentResult)
    assert tuple(step.rule for step in result.steps) == (
        "Rewrite the improper integral as a limit",
        "Apply the Fundamental Theorem of Calculus",
        "Evaluate the limit",
    )


@pytest.mark.parametrize(
    ("query", "expected_rules"),
    [
        (
            "integrate(1/(x^2 + 1), x, -oo, oo)",
            (
                "Split the integral at zero",
                "Use symmetry of an even integrand",
                "Rewrite the improper integral as a limit",
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the limit",
            ),
        ),
        (
            "integrate(x*exp(-x^2), x, -oo, oo)",
            (
                "Split the integral at zero",
                "Use symmetry of an odd integrand",
                "Evaluate one tail as a limit",
                "Cancel the symmetric tails",
            ),
        ),
    ],
)
def test_two_sided_infinite_integrals_use_symmetry(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
) -> None:
    """Whole-line integrals should split at zero and justify symmetry explicitly."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != _OPAQUE_RULE for step in result.steps)
