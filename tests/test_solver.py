"""End-to-end symbolic solver tests."""

import pytest

from stepsolver import ExactResult, Solver, UnsolvedResult, format_ascii

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
        ("diff(sin(x)*exp(x),x)", "Result: (sin(x) + cos(x)) * exp(x)"),
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
            ("Apply the quadratic formula", "Simplify the roots"),
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
        "Use the arctangent integral",
        "Simplify the antiderivative",
    )
    assert all(step.before != step.after for step in result.steps)
    assert result.steps[1].verification.method.value == "differentiation"


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
