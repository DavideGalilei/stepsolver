"""Numerical root-finding regressions for the public ``nsolve`` API."""

import json
import math

import pytest

from stepsolver import (
    ExactResult,
    NoSolutionValue,
    NumericResult,
    NumericRootsValue,
    Operation,
    QueryError,
    Solver,
    UnsolvedResult,
    format_ascii,
    format_latex_value,
    parse,
    solve_isolated,
    solve_payload,
    validate_solve_request,
)
from stepsolver.ast import ApproximateNumber, FunctionCall
from stepsolver.browser import solve_mathjson_json
from stepsolver.web import SolveResponse

_RESIDUAL_TOLERANCE = 1e-8
_MINIMUM_ITERATES = 3
_TWO_REAL_ROOTS = 2


def _single_root(result: NumericResult) -> float:
    assert isinstance(result.value, NumericRootsValue)
    assert len(result.value.roots) == 1
    root = result.value.roots[0]
    assert isinstance(root, ApproximateNumber)
    return float(root.text)


def test_nsolve_k03_returns_verified_newton_trace() -> None:
    """K03 should use the supplied guess and show genuine Newton iterates."""
    result = Solver().nsolve("cos(x) = x", var="x", guess=0.7)

    assert isinstance(result, NumericResult)
    root = _single_root(result)
    assert root == pytest.approx(0.7390851332, abs=1e-8)
    assert result.value.tolerance == pytest.approx(1e-12)
    assert result.residuals
    assert max(abs(residual) for residual in result.residuals) < _RESIDUAL_TOLERANCE
    assert "newton" in result.method.lower()

    apply_step = next(step for step in result.steps if "newton" in step.rule.lower())
    assert isinstance(apply_step.after, FunctionCall)
    assert str(apply_step.after.name) == "newton_iterations"
    iterates = [
        argument
        for argument in apply_step.after.arguments[1:]
        if isinstance(argument, ApproximateNumber)
    ]
    assert len(iterates) >= _MINIMUM_ITERATES
    values = [float(iterate.text) for iterate in iterates]
    assert values[0] == pytest.approx(0.7)
    assert values[-1] == pytest.approx(root, abs=1e-8)

    first, second = values[0], values[1]
    expected_second = first - (math.cos(first) - first) / (-math.sin(first) - 1)
    assert second == pytest.approx(expected_second, abs=1e-8)
    assert any("newton" in note.label.lower() for note in apply_step.notes)

    final_step = result.steps[-1]
    assert "residual" in final_step.rule.lower()
    assert isinstance(final_step.after, FunctionCall)
    assert str(final_step.after.name) == "approximate_solutions"
    final_value = final_step.after.arguments[1]
    assert isinstance(final_value, ApproximateNumber)
    assert float(final_value.text) == pytest.approx(root, abs=1e-12)

    output = format_ascii(result)
    assert "Result: 0.7390851332" in output
    assert "Newton" in output

    payload = solve_payload(result)
    assert payload.status == "numeric"
    assert payload.result_latex is not None
    assert r"\approx" in payload.result_latex
    assert "0.7390851332" in payload.result_latex
    assert format_latex_value(result.value).startswith("0.7390851332")


def test_nsolve_k04_finds_the_real_root_without_a_guess() -> None:
    """K04 should isolate the real root and refine it with Newton's method."""
    result = Solver().nsolve("x^5 - x - 1 = 0", var="x")

    assert isinstance(result, NumericResult)
    assert len(result.value.roots) == 1
    root = _single_root(result)
    assert root == pytest.approx(1.1673039782614187, abs=1e-8)
    assert result.method.lower().startswith("polynomial")
    assert any("isolat" in step.rule.lower() for step in result.steps)
    assert any("newton" in step.rule.lower() for step in result.steps)
    assert result.residuals
    assert max(abs(residual) for residual in result.residuals) < _RESIDUAL_TOLERANCE
    assert "1.167304015" not in format_ascii(result)


def test_nsolve_requires_a_guess_for_a_general_transcendental_equation() -> None:
    """A missing guess should be a typed failure, not an invented default root."""
    result = Solver().nsolve("cos(x) = x", var="x")

    assert isinstance(result, UnsolvedResult)
    assert "initial guess" in result.reason.lower()
    assert result.steps == ()


def test_nsolve_zero_derivative_start_is_a_typed_failure() -> None:
    """A stationary starting point should stop safely with an explanation."""
    result = Solver().nsolve("x^2 + 1 = 0", var="x", guess=0.0)

    assert isinstance(result, UnsolvedResult)
    assert "derivative is zero" in result.reason.lower()


def test_nsolve_nonconvergence_is_bounded_and_typed() -> None:
    """A pathological start should report bounded nonconvergence, not raise."""
    result = Solver().nsolve("cos(x) = x", var="x", guess=1e300)

    assert isinstance(result, UnsolvedResult)
    assert "did not converge" in result.reason.lower()


def test_nsolve_identity_residual_is_a_typed_failure() -> None:
    """An identity has no isolated numerical root to report."""
    result = Solver().nsolve("x = x", var="x", guess=0.0)

    assert isinstance(result, UnsolvedResult)
    assert "identically zero" in result.reason.lower()


def test_nsolve_residual_without_the_variable_is_a_typed_failure() -> None:
    """A constant residual cannot be solved for the selected variable."""
    result = Solver().nsolve("2 = 0", var="x", guess=0.0)

    assert isinstance(result, UnsolvedResult)
    assert "does not depend on the selected variable" in result.reason.lower()


@pytest.mark.parametrize(
    ("equation", "guess", "reason"),
    [
        ("x^2 + y = 0", 1.0, "one real variable"),
        ("sqrt(x) = 2", -1.0, "undefined or non-real"),
        ("1/x = 0", 0.0, "undefined or non-real"),
        ("x^21 - 1 = 0", None, "degree 21 is too high"),
    ],
)
def test_nsolve_rejects_unsafe_numeric_problems(
    equation: str,
    guess: float | None,
    reason: str,
) -> None:
    """Unsupported, non-real, pole, and oversized cases should fail safely."""
    result = Solver().nsolve(equation, var="x", guess=guess)

    assert isinstance(result, UnsolvedResult)
    assert reason in result.reason.lower()


def test_nsolve_initial_guess_already_at_the_root_needs_no_update() -> None:
    """An exact starting root should be reported without a decorative iteration."""
    result = Solver().nsolve("x - 1 = 0", var="x", guess=1.0)

    assert isinstance(result, NumericResult)
    assert _single_root(result) == pytest.approx(1.0)
    assert tuple(step.rule for step in result.steps) == ("Check the initial guess",)
    assert result.residuals[0] == pytest.approx(0.0)


def test_nsolve_polynomial_fallback_reports_isolation_without_a_trace() -> None:
    """A repeated root that stalls Newton should still return the isolated root."""
    result = Solver().nsolve("x^3 = 0", var="x")

    assert isinstance(result, NumericResult)
    assert _single_root(result) == pytest.approx(0.0)
    assert result.method == "polynomial real-root isolation"
    assert tuple(step.rule for step in result.steps) == ("Isolate the real roots",)


def test_nsolve_multiple_real_roots_are_all_reported() -> None:
    """Real-root isolation should return every real root and refine the first."""
    result = Solver().nsolve("x^2 - 4 = 0", var="x")

    assert isinstance(result, NumericResult)
    assert len(result.value.roots) == _TWO_REAL_ROOTS
    assert [
        float(root.text) for root in result.value.roots if isinstance(root, ApproximateNumber)
    ] == (pytest.approx([-2.0, 2.0]))
    assert "[-2, 2]" in format_ascii(result)
    assert any(step.rule == "Apply Newton's method" for step in result.steps)


def test_nsolve_rejects_complex_roots_for_the_real_domain() -> None:
    """A polynomial with only non-real roots should report no real solution."""
    result = Solver().nsolve("x^2 + 1 = 0", var="x")

    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert any("no real roots" in step.explanation.lower() for step in result.steps)


def test_nsolve_polynomial_no_guess_keeps_only_real_roots() -> None:
    """Real-root isolation must not return the complex cube roots of unity."""
    result = Solver().nsolve("x^3 - 1 = 0", var="x")

    assert isinstance(result, NumericResult)
    assert len(result.value.roots) == 1
    assert _single_root(result) == pytest.approx(1.0, abs=1e-8)
    assert "complex roots are excluded" in format_ascii(result).lower()


def test_nsolve_accepts_an_equation_or_a_residual_expression() -> None:
    """Equation and residual forms should reach the same numerical root."""
    equation_result = Solver().nsolve("cos(x) = x", var="x", guess=0.7)
    residual_result = Solver().nsolve("cos(x) - x", var="x", guess=0.7)

    assert isinstance(equation_result, NumericResult)
    assert isinstance(residual_result, NumericResult)
    assert _single_root(equation_result) == pytest.approx(_single_root(residual_result), abs=1e-12)
    assert residual_result.steps[0].rule == "Form the residual equation"


def test_nsolve_first_class_query_is_supported() -> None:
    """The parser and Solver.solve should both understand an nsolve query."""
    query = parse("nsolve(cos(x) = x, x, 0.7)")

    assert query.operation is Operation.NSOLVE
    result = Solver().solve(query)
    assert isinstance(result, NumericResult)
    assert _single_root(result) == pytest.approx(0.7390851332, abs=1e-8)
    assert isinstance(Solver().nsolve(query), NumericResult)


def test_nsolve_convenience_api_validates_its_inputs() -> None:
    """Bad convenience calls should raise typed query errors before solving."""
    with pytest.raises(QueryError):
        Solver().nsolve("cos(x) = x")
    with pytest.raises(QueryError):
        Solver().nsolve("cos(x) = x", var="x + 1")
    with pytest.raises(QueryError):
        Solver().nsolve("cos(x) = x", var="x", guess=float("inf"))
    with pytest.raises(QueryError):
        Solver().nsolve(parse("solve(x^2 - 4 = 0, x)"))


def test_numeric_result_is_available_through_isolated_serialization() -> None:
    """The bounded agent-facing adapter should carry numeric status and latex."""
    request = validate_solve_request("nsolve(cos(x) = x, x, 0.7)")
    solution = solve_isolated(request)

    assert solution.status == "numeric"
    assert solution.result is not None
    assert "0.7390851332" in solution.result
    assert solution.latex is not None
    assert r"\approx" in solution.latex
    assert solution.reason is None


def test_numeric_result_is_serialized_by_the_web_adapter() -> None:
    """MathJSON and the web response should carry the numeric status."""
    source = ["nsolve", ["Equal", ["Cos", "x"], "x"], "x", 0.7]
    payload = SolveResponse.model_validate_json(solve_mathjson_json(json.dumps(source)))

    assert payload.status == "numeric"
    assert payload.result_latex is not None
    assert r"\approx" in payload.result_latex
    assert [step.rule for step in payload.steps] == [
        "Form the residual equation",
        "Apply Newton's method",
        "Check the numerical residual",
    ]


def test_symbolic_solve_behavior_is_unchanged() -> None:
    """The new numeric API must not redirect ordinary symbolic solving."""
    result = Solver().solve("solve(x^2 - 4 = 0, x)")

    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Factor the quadratic",
        "Apply the zero-product property",
        "Solve each factor",
    )
