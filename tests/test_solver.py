"""End-to-end symbolic solver tests."""

import pytest

from stepsolver import (
    ExactResult,
    Solver,
    UnsolvedResult,
    format_ascii,
    format_latex_expression,
)

_CONTOUR_STEP_COUNT = 2


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2+2", "Result: 4"),
        ("simplify((x^2-1)/(x-1))", "Result: x + 1"),
        ("expand((x+1)^3)", "Result: x ^ 3 + 3 * x ^ 2 + 3 * x + 1"),
        ("factor(x^2-4)", "Result: (x - 2) * (x + 2)"),
        ("diff(sin(x)*exp(x),x)", "Result: exp(x) * sin(x) + exp(x) * cos(x)"),
        ("integrate(sin(x),x,0,pi)", "Result: 2"),
        ("limit(sin(x)/x,x,0)", "Result: 1"),
        ("sum(k,k,1,10)", "Result: 55"),
        ("product(k,k,1,5)", "Result: 120"),
        ("det(matrix([[1,2],[3,4]]))", "Result: -2"),
        ("rank(matrix([[1,2],[2,4]]))", "Result: 1"),
        ("gcd(84,30)", "Result: 6"),
        ("lcm(12,18)", "Result: 36"),
        ("is_prime(104729)", "Result: true"),
        ("prime_factors(360)", "Result: {2: 3, 3: 2, 5: 1}"),
        ("binomial(10,3)", "Result: 120"),
        ("permutations(6,2)", "Result: 30"),
        ("combinations(6,2)", "Result: 15"),
    ],
)
def test_supported_queries_have_exact_results(
    solver: Solver,
    query: str,
    expected: str,
) -> None:
    """Representative operations should produce typed exact values and steps."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert result.steps
    assert expected in format_ascii(result)


def test_equation_system_returns_typed_mappings(solver: Solver) -> None:
    """Equation systems should return structured solution mappings."""
    result = solver.solve("solve([x+y=3,x-y=1],[x,y])")
    assert isinstance(result, ExactResult)
    assert "Result: [{x: 2, y: 1}]" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "rules"),
    [
        (
            "solve(2*x+3=7,x)",
            ("Collect variable terms", "Divide by the coefficient"),
        ),
        (
            "solve(x^2-4=0,x)",
            (
                "Factor the quadratic",
                "Apply the zero-product property",
                "Solve each factor",
            ),
        ),
        (
            "solve(x^2+x-1=0,x)",
            ("Apply the quadratic formula",),
        ),
        (
            "solve((x+1)/(x-2)=3,x)",
            (
                "Clear the denominators",
                "Collect variable terms",
                "Divide by the coefficient",
            ),
        ),
    ],
)
def test_equations_have_verified_detailed_derivations(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
) -> None:
    """Common equation families should expose real algebraic transformations."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert all(step.before != step.after for step in result.steps)
    assert all(
        step.verification.method.value == "solution-set equivalence" for step in result.steps
    )


def test_unsupported_derivation_has_an_honest_fallback(solver: Solver) -> None:
    """Unsupported equation families should not invent a fake worked derivation."""
    result = solver.solve("solve(x^3-1=0,x)")
    assert isinstance(result, ExactResult)
    assert len(result.steps) == 1
    assert result.steps[0].rule == "Compute exact result"


def test_reciprocal_quadratic_integral_has_worked_steps(solver: Solver) -> None:
    """A completed-square integral should expose its arctangent derivation."""
    result = solver.solve("integrate(1/(x^2-x+1),x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Complete the square",
        "Substitute to get a unit denominator",
        "Use the basic arctangent rule",
        "Substitute back",
    )
    assert all(step.before != step.after for step in result.steps)
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Take half the linear coefficient, then square it",
        "Add and subtract that number",
        "Recognize the perfect square",
        "General pattern",
    )
    assert result.steps[1].verification.method.value == "substitution"
    assert tuple(note.label for note in result.steps[1].notes) == (
        "Choose the substitution",
        "Rewrite the shifted term",
        "Change the differential",
    )
    assert result.steps[2].verification.method.value == "differentiation"
    assert tuple(note.label for note in result.steps[2].notes) == ("Rule to remember",)
    assert "u²" not in result.steps[2].explanation
    assert format_ascii(result).endswith("+ C")


def test_dirichlet_integral_has_a_parameter_derivation(solver: Solver) -> None:
    """The classical improper integral should explain the damping-parameter method."""
    result = solver.solve("integrate(sin(x)/x,x,0,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    )
    assert result.steps[1].verification.method.value == "differentiation"
    assert tuple(note.label for note in result.steps[3].notes) == (
        "The damped integral vanishes",
        "Arctangent limit",
        "Therefore",
    )
    assert format_ascii(result).endswith("pi / 2")


@pytest.mark.parametrize(
    ("query", "rules"),
    [
        (
            "solve((x-1)^2=0,x)",
            ("Set the repeated factor equal to zero", "Solve each factor"),
        ),
        (
            "solve(2*x^2+4*x+2=0,x)",
            (
                "Factor the quadratic",
                "Set the repeated factor equal to zero",
                "Solve each factor",
            ),
        ),
        (
            "solve(x^2+1=0,x)",
            ("Calculate the discriminant", "Conclude there are no real solutions"),
        ),
        (
            "solve(x=x+1,x)",
            ("Simplify the equation", "Conclude there are no solutions"),
        ),
        ("integrate(sin(x),x)", ("Use the sine antiderivative",)),
        (
            "integrate(sin(2*x),x)",
            ("Apply the reverse chain rule for sine",),
        ),
        ("integrate(x^2,x)", ("Use the power rule",)),
        (
            "integrate(x/(x^2+1),x)",
            ("Substitute the denominator", "Use the logarithm rule", "Substitute back"),
        ),
        ("integrate(1/(x^2+1),x)", ("Use the basic arctangent rule",)),
        (
            "integrate(1/(4*x^2+1),x)",
            (
                "Substitute to get a unit denominator",
                "Use the basic arctangent rule",
                "Substitute back",
            ),
        ),
    ],
)
def test_human_first_edge_case_derivations(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
) -> None:
    """Degenerate and direct cases should use the shortest meaningful method."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert all(step.rule != "Compute exact result" for step in result.steps)


def test_derivations_hide_neutral_algebra_artifacts(solver: Solver) -> None:
    """Student-facing transformations should not display identity arithmetic."""
    direct = solver.solve("integrate(1/(x^2+1),x)")
    scaled = solver.solve("integrate(1/(4*x^2+1),x)")
    logarithmic = solver.solve("integrate(x/(x^2+1),x)")
    assert isinstance(direct, ExactResult)
    assert isinstance(scaled, ExactResult)
    assert isinstance(logarithmic, ExactResult)
    rendered = " ".join(
        format_latex_expression(expression)
        for result in (direct, scaled, logarithmic)
        for step in result.steps
        for expression in (step.before, step.after)
    )
    assert "1 \\cdot" not in rendered
    assert r"\frac{2}{2}" not in rendered
    assert r"\frac{\frac{1}{2}}{1}" not in rendered
    positivity = logarithmic.steps[1].notes[0].expression
    assert format_latex_expression(positivity) == r"x^{2} + 1 > 0"


@pytest.mark.parametrize("query", ["solve(x=x,x)", "solve((x-1)/(x-1)=1,x)"])
def test_identity_equations_do_not_return_a_false_empty_set(
    solver: Solver,
    query: str,
) -> None:
    """Universal identities should remain honest until domain sets are representable."""
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert "identity" not in result.reason.lower() or "domain" in result.reason.lower()
    assert "universal solution set" in result.reason


def test_no_real_solution_is_rendered_as_the_empty_set(solver: Solver) -> None:
    """A negative discriminant should end with mathematical empty-set notation."""
    result = solver.solve("solve(x^2+1=0,x)")
    assert isinstance(result, ExactResult)
    assert format_latex_expression(result.steps[-1].after) == r"\varnothing"


def test_scaled_dirichlet_integral_reduces_to_the_standard_case(solver: Solver) -> None:
    """A positive sine frequency should be removed by an explicit scaling substitution."""
    result = solver.solve("integrate(sin(2*x)/x,x,0,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Scale the integration variable",
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    )
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Choose the substitution",
        "Rewrite the original variable",
        "Change the differential",
    )
    assert format_ascii(result).endswith("pi / 2")


def test_inequality_solver(solver: Solver) -> None:
    """A real univariate inequality should produce an exact interval condition."""
    result = solver.solve("solve_inequality(x^2<4,x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert "-2 < x" in rendered
    assert "x < 2" in rendered


def test_parameterized_unit_circle_integral(solver: Solver) -> None:
    """The positively oriented unit circle integral of 1/z should be 2*pi*i."""
    result = solver.solve("contour_integrate(1/z,z,exp(i*t),t,0,2*pi)")
    assert isinstance(result, ExactResult)
    assert len(result.steps) == _CONTOUR_STEP_COUNT
    assert "Result: 2 * i * pi" in format_ascii(result)


def test_reversed_contour_reverses_sign(solver: Solver) -> None:
    """Reversing the unit-circle parameter bounds should reverse orientation."""
    result = solver.solve("contour_integrate(1/z,z,exp(i*t),t,2*pi,0)")
    assert isinstance(result, ExactResult)
    assert "Result: -2 * i * pi" in format_ascii(result)


def test_matrix_inverse_is_structured(solver: Solver) -> None:
    """Matrix-valued answers should remain structured StepSolver values."""
    result = solver.solve("inverse(matrix([[1,2],[3,4]]))")
    assert isinstance(result, ExactResult)
    assert "Result: matrix([[-2, 1], [3/2, -1/2]])" in format_ascii(result)


def test_numeric_is_explicit(solver: Solver) -> None:
    """Numerical approximation should require the numeric operation."""
    result = solver.solve("numeric(pi,8)")
    assert isinstance(result, ExactResult)
    assert "Result: 3.1415927" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cancel((x^2-1)/(x-1))", "Result: x + 1"),
        ("apart(1/(x*(x+1)),x)", "Result: -1 / (x + 1) + 1 / x"),
        ("diff(x^4,x,2)", "Result: 12 * x ^ 2"),
        ("diff(log(x^2+1),x)", "Result: 2 * x / (x ^ 2 + 1)"),
        ("integrate(x^2,x)", "Result: x ^ 3 / 3"),
        ("limit(1/x,x,0,right)", "Result: oo"),
        ("series(exp(x),x,0,4)", "Result: 1 + x + x ^ 2 / 2 + x ^ 3 / 6 + O(x ^ 4)"),
        ("matrix([[1,2],[3,4]])", "Result: matrix([[1, 2], [3, 4]])"),
        ("rref(matrix([[1,2],[2,4]]))", "Result: [matrix([[1, 2], [0, 0]]), [0]]"),
        ("eigenvalues(matrix([[2,0],[0,3]]))", "Result: {2: 1, 3: 1}"),
        ("dsolve(diff(y(x),x)=y(x),y(x))", "Result: Eq(y(x), C1 * exp(x))"),
        ("rsolve(a(n+1)-a(n)=1,a(n))", "Result: C0 + n"),
        ("laplace(exp(-t),t,s)", "Result: 1 / (s + 1)"),
        ("inverse_laplace(1/(s+1),s,t)", "Result: exp(-t) * Heaviside(t)"),
        ("fourier(exp(-pi*x^2),x,k)", "Result: exp(-pi * k ^ 2)"),
        ("inverse_fourier(exp(-pi*k^2),k,x)", "Result: exp(-pi * x ^ 2)"),
        ("numeric(sqrt(2))", "Result: 1.41421356237310"),
    ],
)
def test_extended_symbolic_operations(solver: Solver, query: str, expected: str) -> None:
    """The broader calculus, matrix, transform, and recurrence surface should work."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert expected in format_ascii(result)


@pytest.mark.parametrize(
    "query",
    ["integrate(x)", "matrix([])", "is_prime(x)", "limit(x,x,0,sideways)"],
)
def test_invalid_operation_arguments_return_unsolved(solver: Solver, query: str) -> None:
    """Semantically invalid queries should return a typed partial result."""
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert result.reason
