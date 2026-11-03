"""Native identity, numerical solving, and ODE syntax regressions."""

from __future__ import annotations

import math

import pytest

from stepsolver import (
    ExactResult,
    NumericResult,
    NumericRootsValue,
    Operation,
    Solver,
    UniversalSolutionValue,
    UnsolvedResult,
    format_ascii,
    format_expression,
    parse,
    parse_expression,
    solve_isolated,
    validate_solve_request,
)
from stepsolver.ast import (
    BinaryExpression,
    Constant,
    ConstantName,
    FunctionCall,
    SequenceExpression,
)

_TUPLE_ARGUMENT_LENGTH = 2
_RESIDUAL_TOLERANCE = 1e-8
_ROOT_TOLERANCE = 1e-12


@pytest.fixture
def solver() -> Solver:
    """Return a fresh solver for each regression."""
    return Solver()


def test_identity_equation_serializes_as_a_universal_solution() -> None:
    """An identity should survive both the typed value and isolated serialization."""
    result = Solver().solve("solve((x + 1)^2 = x^2 + 2*x + 1, x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, UniversalSolutionValue)
    assert result.value.exclusions == ()

    solution = solve_isolated(validate_solve_request("solve((x + 1)^2 = x^2 + 2*x + 1, x)"))
    assert solution.status == "exact"
    assert solution.result == "all real numbers"
    assert solution.latex is not None
    assert r"\mathbb{R}" in solution.latex


def test_identity_with_a_denominator_keeps_its_exclusion() -> None:
    """The universal solution must not swallow the original domain restriction."""
    result = Solver().solve("solve(1/(x - 1) = 1/(x - 1), x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, UniversalSolutionValue)
    assert tuple(format_expression(item) for item in result.value.exclusions) == ("1",)

    solution = solve_isolated(validate_solve_request("solve(1/(x - 1) = 1/(x - 1), x)"))
    assert solution.result == "all real numbers except 1"


def test_contradiction_is_not_reported_as_a_universal_solution(solver: Solver) -> None:
    """Matching variable terms with unequal constants remain a contradiction."""
    result = solver.solve("solve(2*x + 1 = 2*x + 2, x)")
    assert isinstance(result, ExactResult)
    assert not isinstance(result.value, UniversalSolutionValue)
    assert "No solution" in format_ascii(result)


def test_parenthesized_derivative_order_parses_as_a_sequence() -> None:
    """SymPy's ``(variable, order)`` argument should become a public sequence."""
    expression = parse_expression("Derivative(y(x), (x, 2))")
    assert isinstance(expression, FunctionCall)
    assert str(expression.name) == "Derivative"
    assert isinstance(expression.arguments[1], SequenceExpression)
    assert len(expression.arguments[1].items) == _TUPLE_ARGUMENT_LENGTH


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("dsolve(Derivative(y(x), x) + 2*y(x) = 0, y(x))", "Eq(y(x), C1 * exp(-2 * x))"),
        (
            "dsolve(Derivative(y(x), (x, 2)) - 3*Derivative(y(x), x) + 2*y(x) = 0, y(x))",
            "Eq(y(x), (C1 + C2 * exp(x)) * exp(x))",
        ),
        ("dsolve(Derivative(y(x), x) - 3*y(x) = 0, y(x))", "Eq(y(x), C1 * exp(3 * x))"),
        (
            "dsolve(Derivative(y(x), (x, 2)) + y(x) = 0, y(x))",
            "Eq(y(x), C1 * sin(x) + C2 * cos(x))",
        ),
    ],
)
def test_dsolve_accepts_sympy_derivative_notation(
    solver: Solver,
    query: str,
    expected: str,
) -> None:
    """SymPy-style derivative syntax should produce the same verified ODE families."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith(f"Result: {expected}")
    assert tuple(step.rule for step in result.steps) == (
        "Identify the differential equation",
        "Apply the classified solution method",
        "Check the solution by substitution",
    )


def test_native_tuple_derivative_order_matches_the_three_argument_form(solver: Solver) -> None:
    """``diff(y(x), (x, 2))`` and ``diff(y(x), x, 2)`` should agree."""
    tuple_form = solver.solve("dsolve(diff(y(x), (x, 2)) + y(x) = 0, y(x))")
    order_form = solver.solve("dsolve(diff(y(x), x, 2) + y(x) = 0, y(x))")
    assert isinstance(tuple_form, ExactResult)
    assert isinstance(order_form, ExactResult)
    assert (
        format_ascii(tuple_form).rsplit("Result:", maxsplit=1)[-1]
        == format_ascii(order_form).rsplit("Result:", maxsplit=1)[-1]
    )


def test_malformed_derivative_tuple_is_a_typed_failure(solver: Solver) -> None:
    """An invalid derivative order must not fabricate a solution family."""
    result = solver.solve("dsolve(Derivative(y(x), (x, 1, 2)) = 0, y(x))")
    assert isinstance(result, UnsolvedResult)
    assert "tuple expects a variable and an order" in result.reason


def test_derivative_notation_is_available_inside_a_simplification(solver: Solver) -> None:
    """The derivative name should convert normally outside an ODE query too."""
    result = solver.solve("simplify(Derivative(x^3, (x, 2)))")
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith("Result: 6 * x")


@pytest.mark.parametrize(
    ("query", "expected_root"),
    [
        ("nsolve(cos(x) = x, x, 0.7)", 0.7390851332),
        ("nsolve(x^3 - 2*x - 5 = 0, x, 2)", 2.0945514815423),
        ("nsolve(exp(-x) = x, x, 0.5)", 0.5671432904098),
        ("nsolve(sin(x) = 1/2, x, 0.5)", 0.5235987755983),
        ("nsolve(ln(x) = 1, x, 2.5)", math.e),
    ],
)
def test_nsolve_native_queries_return_verified_roots(
    solver: Solver,
    query: str,
    expected_root: float,
) -> None:
    """Every benchmark numerical case should dispatch through the native operation."""
    parsed = parse(query)
    assert parsed.operation is Operation.NSOLVE
    result = solver.solve(parsed)
    assert isinstance(result, NumericResult)
    assert isinstance(result.value, NumericRootsValue)
    assert len(result.value.roots) == 1
    root = float(format_expression(result.value.roots[0]))
    assert root == pytest.approx(expected_root, abs=_RESIDUAL_TOLERANCE)
    assert result.value.tolerance == pytest.approx(_ROOT_TOLERANCE)
    assert result.residuals
    assert max(abs(item) for item in result.residuals) < _RESIDUAL_TOLERANCE
    assert any("newton" in step.rule.lower() for step in result.steps)


def test_nsolve_isolated_serialization_reports_a_numeric_status() -> None:
    """The bounded adapter should preserve the numerical answer and its status."""
    solution = solve_isolated(validate_solve_request("nsolve(cos(x) = x, x, 0.7)"))
    assert solution.status == "numeric"
    assert solution.result is not None
    assert "0.7390851332" in solution.result
    assert solution.latex is not None
    assert r"\approx" in solution.latex


def test_dsolve_isolated_serialization_reports_an_equation() -> None:
    """The bounded adapter should carry the ODE family as an equation."""
    solution = solve_isolated(
        validate_solve_request("dsolve(Derivative(y(x), x) + 2*y(x) = 0, y(x))")
    )
    assert solution.status == "exact"
    assert solution.result == "Eq(y(x), C1 * exp(-2 * x))"
    assert solution.latex is not None


def test_imaginary_guess_is_rejected_by_nsolve(solver: Solver) -> None:
    """A complex initial guess is not a valid real Newton starting point."""
    query = parse("nsolve(cos(x) = x, x, i)")
    guess = query.arguments[2]
    assert isinstance(guess, Constant)
    assert guess.name is ConstantName.IMAGINARY
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert "finite real" in result.reason


def test_derivative_of_a_symbolic_product_is_still_converted(solver: Solver) -> None:
    """The derivative converter should evaluate ordinary symbolic targets."""
    query = parse("simplify(Derivative(x*y, x))")
    equation = parse_expression("Derivative(x*y, x)")
    assert isinstance(equation, FunctionCall)
    assert isinstance(equation.arguments[0], BinaryExpression)
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith("Result: y")
