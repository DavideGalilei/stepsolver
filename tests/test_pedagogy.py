"""Regression matrix for student-first method selection and mathematical display."""

import pytest
import sympy as sp

from stepsolver import (
    ExactResult,
    ScalarValue,
    Solver,
    format_ascii,
    format_expression,
    format_latex_expression,
    format_latex_value,
)
from stepsolver.sympy_conversion import SympyConverter


@pytest.fixture
def solver() -> Solver:
    """Create the symbolic solver used by the pedagogy matrix."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected_rules"),
    [
        ("solve(x^2+x-1=0,x)", ("Apply the quadratic formula",)),
        ("diff(x^5,x)", ("Use the power rule",)),
        ("diff(sin(x^2),x)", ("Apply the chain rule",)),
        ("diff((x^2+1)^3,x)", ("Apply the power and chain rules",)),
        ("diff(x^2+sin(x),x)", ("Differentiate term by term",)),
        ("diff(x^2*sin(x),x)", ("Apply the product rule", "Simplify the derivative")),
        ("diff(sin(x)/x,x)", ("Apply the quotient rule", "Simplify the derivative")),
        (
            "integrate(3*sin(x),x)",
            ("Factor out the constant", "Use the sine antiderivative"),
        ),
        (
            "integrate(2*x*cos(x^2),x)",
            ("Substitute the inner function", "Integrate in the new variable", "Substitute back"),
        ),
        ("integrate(sin(2*x),x)", ("Apply the reverse chain rule for sine",)),
        (
            "integrate(x^3-2*x+4,x)",
            ("Split the integral across the sum", "Integrate each term"),
        ),
        (
            "integrate((2*x+1)/(x^2+x+1),x)",
            ("Substitute the denominator", "Use the logarithm rule", "Substitute back"),
        ),
        (
            "integrate(1/(x^2-x+1),x)",
            (
                "Complete the square",
                "Substitute to get a unit denominator",
                "Use the basic arctangent rule",
                "Substitute back",
            ),
        ),
        (
            "integrate(1/(sqrt(x)*(x+1)),x)",
            ("Substitute the square root", "Use the arctangent rule", "Substitute back"),
        ),
        (
            "integrate(x*exp(x),x)",
            ("Choose integration by parts", "Evaluate the remaining integral"),
        ),
        (
            "integrate(sin(x)^2,x)",
            ("Use the sine power-reduction identity", "Integrate the reduced expression"),
        ),
        (
            "integrate(cos(x)^2,x)",
            ("Use the cosine power-reduction identity", "Integrate the reduced expression"),
        ),
        (
            "integrate(1/(4*x^2+1),x)",
            (
                "Substitute to get a unit denominator",
                "Use the basic arctangent rule",
                "Substitute back",
            ),
        ),
        ("integrate(exp(-x^2),x)", ("Express the antiderivative with the error function",)),
        (
            "integrate(1/(x^2*(x^2+25)),x)",
            ("Decompose into partial fractions", "Integrate each partial fraction"),
        ),
        (
            "integrate(1/sqrt(25*x^2+2),x)",
            (
                "Normalize the quadratic radical",
                "Use the inverse hyperbolic sine rule",
                "Substitute back",
            ),
        ),
        (
            "integrate(sqrt(2*x-x^2),x)",
            (
                "Complete the square under the radical",
                "Shift the variable",
                "Use the semicircle antiderivative",
                "Substitute back",
            ),
        ),
        (
            "integrate(x^2,x,0,2)",
            (
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the bounds",
                "Finish the arithmetic",
            ),
        ),
        (
            "integrate(exp(-x),x,0,oo)",
            (
                "Rewrite the improper integral as a limit",
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the limit",
            ),
        ),
        (
            "integrate(exp(x),x,-oo,0)",
            (
                "Rewrite the improper integral as a limit",
                "Apply the Fundamental Theorem of Calculus",
                "Evaluate the limit",
            ),
        ),
        (
            "limit(sin(x)/x,x,0)",
            (
                "Check direct substitution",
                "Apply L'Hôpital's rule",
                "Substitute into the transformed limit",
            ),
        ),
        (
            "limit(sin(3*x)/x,x,0)",
            (
                "Check direct substitution",
                "Apply L'Hôpital's rule",
                "Substitute into the transformed limit",
            ),
        ),
        (
            "limit((x^2-1)/(x-1),x,1)",
            ("Factor and cancel the common factor", "Substitute into the simplified expression"),
        ),
        ("limit(1/x,x,0,right)", ("Analyze the sign from the requested side",)),
        ("limit(1/x,x,0,left)", ("Analyze the sign from the requested side",)),
        ("limit((3*x^2+1)/(2*x^2-5),x,oo)", ("Compare the leading powers",)),
        ("limit(exp(x)/x^3,x,oo)", ("Use exponential growth",)),
        ("limit(cos(x),x,0)", ("Use direct substitution",)),
        (
            "limit(x*sin(1/x),x,0)",
            (
                "Bound the oscillating factor",
                "Multiply the bounds by the vanishing factor",
                "Apply the squeeze theorem",
            ),
        ),
    ],
)
def test_common_families_choose_the_expected_human_method(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
) -> None:
    """The first explanation should be the method a student would normally choose."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != "Compute exact result" for step in result.steps)


@pytest.mark.parametrize(
    ("query", "oscillating"),
    [
        ("limit(x*sin(1/x),x,0)", "sin(1 / x)"),
        ("limit(x*cos(1/x),x,0)", "cos(1 / x)"),
        ("limit(x^2*sin(1/x),x,0)", "sin(1 / x)"),
    ],
)
def test_bounded_oscillation_limit_uses_the_squeeze_theorem(
    solver: Solver,
    query: str,
    oscillating: str,
) -> None:
    """A vanishing factor times a bounded oscillation must show both squeeze bounds."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Bound the oscillating factor",
        "Multiply the bounds by the vanishing factor",
        "Apply the squeeze theorem",
    )
    bounding_step, multiplication_step, squeeze_step = result.steps
    assert len(bounding_step.introduced_constraints) == 1
    assert format_expression(bounding_step.introduced_constraints[0].expression).endswith("!= 0")
    multiplication_expression = format_expression(multiplication_step.after)
    assert oscillating in multiplication_expression
    assert "<=" in multiplication_expression
    labels = {note.label for note in squeeze_step.notes}
    assert {
        "Lower product bound",
        "Upper product bound",
        "Bounding limit",
    } <= labels
    assert any(oscillating in format_expression(note.expression) for note in squeeze_step.notes)
    assert any("<=" in format_expression(note.expression) for note in squeeze_step.notes)


def test_student_facing_latex_has_no_backend_or_neutral_artifacts(solver: Solver) -> None:
    """Worked mathematics should not expose serialization or identity arithmetic."""
    queries = (
        "diff(sin(x)*exp(x),x)",
        "integrate(x*sin(x),x)",
        "integrate(sin(x)^2,x)",
        "integrate(sin(x),x,0,pi)",
        "limit(sin(3*x)/x,x,0)",
    )
    rendered: list[str] = []
    for query in queries:
        result = solver.solve(query)
        assert isinstance(result, ExactResult)
        rendered.extend(
            format_latex_expression(expression)
            for step in result.steps
            for expression in (step.before, step.after)
        )
        rendered.extend(
            format_latex_expression(note.expression) for step in result.steps for note in step.notes
        )
    joined = " ".join(rendered)
    assert "InvisibleOperator" not in joined
    assert r"\operatorname{Derivative}" not in joined
    assert r"1 \cdot \mathrm{d}" not in joined
    assert "+ -" not in joined
    assert "- -" not in joined
    assert "C +" not in joined


def test_general_power_rule_uses_prime_notation_without_backend_leaks(
    solver: Solver,
) -> None:
    """The logarithmic-differentiation identity should look like textbook mathematics."""
    result = solver.solve("diff(x^x,x)")
    assert isinstance(result, ExactResult)
    rule = format_latex_expression(result.steps[0].notes[0].expression)
    assert r"g'\left(x\right)" in rule
    assert r"f'\left(x\right)" in rule
    assert "Derivative" not in rule
    assert r"\operatorname" not in rule


def test_integration_by_parts_explains_each_choice_separately(solver: Solver) -> None:
    """The learner should see u, dv, du, and v rather than an opaque tuple."""
    result = solver.solve("integrate(x*exp(x),x)")
    assert isinstance(result, ExactResult)
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Integration by parts",
        "Choose the algebraic part",
        "Choose the remaining differential",
        "Differentiate u",
        "Antidifferentiate dv",
    )
    assert format_latex_expression(result.steps[0].notes[0].expression) == (
        r"\int u\,\mathrm{d}v = uv - \int v\,\mathrm{d}u"
    )


def test_definite_integral_keeps_upper_minus_lower_visible(solver: Solver) -> None:
    """Endpoint substitution and signed arithmetic should be explicit."""
    result = solver.solve("integrate(sin(x),x,0,pi)")
    assert isinstance(result, ExactResult)
    assert format_latex_expression(result.steps[0].after) == (
        r"\left[-\cos\left(x\right)\right]_{0}^{\pi}"
    )
    assert format_latex_expression(result.steps[1].after) == r"1 - \left(-1\right)"
    assert tuple(note.label for note in result.steps[1].notes) == (
        "Upper bound",
        "Lower bound",
    )


def test_quadratic_formula_keeps_fraction_structure(solver: Solver) -> None:
    """Substitution into the formula should not become multiplication by one half."""
    result = solver.solve("solve(x^2+x-1=0,x)")
    assert isinstance(result, ExactResult)
    rendered = format_latex_expression(result.steps[0].after)
    assert rendered == (
        r"\left[x = \frac{-\sqrt{5} - 1}{2}, "
        r"x = \frac{-1 + \sqrt{5}}{2}\right]"
    )
    assert r"\cdot \left(\frac{1}{2}\right)" not in rendered


def test_reciprocal_integral_uses_absolute_value_and_preserves_the_pole(
    solver: Solver,
) -> None:
    """A reciprocal should integrate to a real logarithm on both sides of zero."""
    result = solver.solve("integrate(1/x,x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, ScalarValue)
    assert tuple(step.rule for step in result.steps) == (
        "Substitute the linear denominator",
        "Use the logarithmic rule for nonzero inputs",
        "Substitute back",
    )
    assert format_ascii(result).endswith("Result: log(abs(x)) + C")
    assert format_latex_value(result.value) == r"\log\left(\left|x\right|\right) + C"
    constraints = result.steps[0].introduced_constraints
    assert tuple(format_expression(item.expression) for item in constraints) == ("x != 0",)
    assert "absolute value" in result.steps[1].explanation
    assert all(step.rule != "Compute exact result" for step in result.steps)


def test_reciprocal_antiderivative_is_real_and_differentiates_away_from_zero(
    solver: Solver,
) -> None:
    """The displayed antiderivative must recover 1/x on both real intervals."""
    result = solver.solve("integrate(1/x,x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, ScalarValue)
    x = sp.Symbol("x", real=True)
    constant = sp.Symbol("C")
    antiderivative = SympyConverter().to_sympy(result.value.expression)
    derivative = sp.diff(antiderivative, x)
    for point in (
        sp.Integer(-4),
        sp.Rational(-1, 2),
        sp.Rational(1, 2),
        sp.Integer(4),
    ):
        assert antiderivative.subs(x, point).subs(constant, sp.Integer(0)).is_real is True
        assert sp.simplify(derivative.subs(x, point) - 1 / point) == sp.Integer(0)


@pytest.mark.parametrize(
    ("query", "expected", "expected_constraints"),
    [
        (
            "integrate(1/(x+1),x)",
            "log(abs(x + 1)) + C",
            ("x + 1 != 0", "x != -1"),
        ),
        (
            "integrate(1/(2*x+1),x)",
            "log(abs(2 * x + 1)) / 2 + C",
            ("2 * x + 1 != 0", "x != -1/2"),
        ),
    ],
)
def test_reciprocal_integral_handles_shifted_linear_denominators(
    solver: Solver,
    query: str,
    expected: str,
    expected_constraints: tuple[str, ...],
) -> None:
    """A shifted reciprocal should retain its coefficient, root, and pole."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith(f"Result: {expected}")
    constraints = result.steps[0].introduced_constraints
    assert tuple(format_expression(item.expression) for item in constraints) == expected_constraints


def test_general_rational_logarithmic_integrals_keep_verified_result_with_poles(
    solver: Solver,
) -> None:
    """A verified partial-fraction result is retained with its real pole exclusions."""
    result = solver.solve("integrate(1/(x^2-1),x)")
    assert isinstance(result, ExactResult)
    assert result.steps
    constraints = result.steps[0].introduced_constraints
    assert tuple(format_expression(item.expression) for item in constraints) == (
        "x != -1",
        "x != 1",
    )
    assert all(step.rule != "Compute exact result" for step in result.steps)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("integrate(3*x^2,x)", "x ^ 3 + C"),
        ("integrate(2*x+5,x)", "x ^ 2 + 5 * x + C"),
        ("integrate(cos(x),x)", "sin(x) + C"),
    ],
)
def test_reciprocal_rule_does_not_change_nearby_integrals(
    solver: Solver,
    query: str,
    expected: str,
) -> None:
    """The reciprocal specialization should not intercept unrelated elementary integrals."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith(f"Result: {expected}")
    assert all(step.rule != "Compute exact result" for step in result.steps)


@pytest.mark.parametrize(
    "query",
    [
        "integrate(x*tan(x)^2,x)",
        "integrate(x^2/sqrt(x^2+25),x)",
        "integrate(x^5*sqrt(2-x^3),x)",
        "integrate(tan(log(x))^3/x,x)",
    ],
)
def test_remaining_tough_integral_benchmarks_need_real_derivations(
    solver: Solver,
    query: str,
) -> None:
    """Advanced benchmark integrals should expose their actual classroom method."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert all(step.rule != "Compute exact result" for step in result.steps)
