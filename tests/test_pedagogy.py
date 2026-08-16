"""Regression matrix for student-first method selection and mathematical display."""

import pytest

from stepsolver import ExactResult, Solver, format_latex_expression


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
        ("limit(sin(x)/x,x,0)", ("Use the standard sine limit",)),
        ("limit(sin(3*x)/x,x,0)", ("Normalize to the standard sine limit",)),
        (
            "limit((x^2-1)/(x-1),x,1)",
            ("Factor and cancel the common factor", "Substitute into the simplified expression"),
        ),
        ("limit(1/x,x,0,right)", ("Analyze the sign from the requested side",)),
        ("limit(1/x,x,0,left)", ("Analyze the sign from the requested side",)),
        ("limit((3*x^2+1)/(2*x^2-5),x,oo)", ("Compare the leading powers",)),
        ("limit(exp(x)/x^3,x,oo)", ("Use exponential growth",)),
        ("limit(cos(x),x,0)", ("Use direct substitution",)),
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
            format_latex_expression(note.expression)
            for step in result.steps
            for note in step.notes
        )
    joined = " ".join(rendered)
    assert "InvisibleOperator" not in joined
    assert r"\operatorname{Derivative}" not in joined
    assert r"1 \cdot \mathrm{d}" not in joined
    assert "+ -" not in joined
    assert "- -" not in joined
    assert "C +" not in joined


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


@pytest.mark.parametrize("query", ["integrate(1/x,x)", "integrate(1/(x^2-1),x)"])
def test_domain_sensitive_logarithmic_integrals_are_not_faked(
    solver: Solver,
    query: str,
) -> None:
    """Real poles require absolute values and interval-aware domain information."""
    result = solver.solve(query)
    assert not isinstance(result, ExactResult)
    assert "absolute values" in result.reason
    assert "domain" in result.reason


@pytest.mark.parametrize(
    "query",
    [
        "integrate(x*tan(x)^2,x)",
        "integrate(x^2/sqrt(x^2+25),x)",
        "integrate(x^5*sqrt(2-x^3),x)",
        "integrate(tan(log(x))^3/x,x)",
    ],
)
@pytest.mark.xfail(
    strict=True,
    reason="Tracked Tough Integrals benchmark family still needs a dedicated human derivation.",
)
def test_remaining_tough_integral_benchmarks_need_real_derivations(
    solver: Solver,
    query: str,
) -> None:
    """Known benchmark gaps must not be mistaken for completed pedagogy coverage."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert all(step.rule != "Compute exact result" for step in result.steps)
