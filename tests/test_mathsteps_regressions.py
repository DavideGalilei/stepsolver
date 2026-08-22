"""Hard algebra regressions adapted from the google/mathsteps test corpus.

The source project is Copyright 2017 Evy Kassirer and licensed under Apache-2.0.
See THIRD_PARTY_NOTICES.md for provenance. Inputs use StepSolver's explicit
multiplication syntax, and assertions target StepSolver's own result and step models.
"""

import pytest

from stepsolver import ExactResult, Solver, format_ascii, format_expression

_MIN_EQUATION_STEPS = 2


@pytest.fixture
def solver() -> Solver:
    """Create the default solver for the imported regression corpus."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected_rule", "expected_result"),
    [
        ("simplify((2+2)*5)", "Evaluate the arithmetic", "Result: 20"),
        (
            "simplify(x^2+3*x*(-4*x)+5*x^3+3*x^2+6)",
            "Combine like terms",
            "Result: 5 * x ^ 3 - 8 * x ^ 2 + 6",
        ),
        (
            "simplify(2*x^2*y*x*y^3)",
            "Combine powers with the same base",
            "Result: 2 * x ^ 3 * y ^ 4",
        ),
        (
            "simplify(x^y*x^z)",
            "Combine powers with the same base",
            "Result: x ^ (y + z)",
        ),
        (
            "simplify(2*x*4*x/2/4)",
            "Combine powers with the same base",
            "Result: x ^ 2",
        ),
        (
            "expand((3+x)*(4+x)*(x+5))",
            "Apply the distributive property",
            "Result: x ^ 3 + 12 * x ^ 2 + 47 * x + 60",
        ),
        (
            "expand((3*x+5)^2)",
            "Apply the distributive property",
            "Result: 9 * x ^ 2 + 30 * x + 25",
        ),
        ("simplify(4/9+3/5)", "Use a common denominator", "Result: 47/45"),
        (
            "simplify(5*x+(1/2)*x)",
            "Combine like terms",
            "Result: 11 * x / 2",
        ),
        (
            "simplify(2*(x+1))",
            "Expand and combine like terms",
            "Result: 2 * x + 2",
        ),
        ("simplify(-x+x)", "Combine like terms", "Result: 0"),
        ("simplify(x^1)", "Apply algebraic identities", "Result: x"),
    ],
)
def test_adapted_simplification_cases_use_recognizable_human_rules(
    solver: Solver,
    query: str,
    expected_rule: str,
    expected_result: str,
) -> None:
    """Difficult simplifications should name the algebra instead of the backend."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (expected_rule,)
    assert result.steps[0].before != result.steps[0].after
    assert format_ascii(result).endswith(expected_result)


@pytest.mark.parametrize(
    ("query", "expected_result", "expected_constraint"),
    [
        (
            "simplify(((2+x)*(3+x))/(2+x))",
            "Result: x + 3",
            "2 + x != 0",
        ),
        (
            "simplify((x^3*y)/x^2+5)",
            "Result: x * y + 5",
            "x ^ 2 != 0",
        ),
    ],
)
def test_adapted_cancellation_cases_retain_original_domain_restrictions(
    solver: Solver,
    query: str,
    expected_result: str,
    expected_constraint: str,
) -> None:
    """Canceled factors must not silently erase excluded input values."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Cancel common factors",)
    assert tuple(
        format_expression(constraint.expression)
        for constraint in result.steps[0].introduced_constraints
    ) == (expected_constraint,)
    assert "wherever the original denominators are nonzero" in (result.steps[0].verification.detail)
    assert format_ascii(result).endswith(expected_result)


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("solve(5*x/2+2=3*x/2+10,x)", "Result: [{x: 8}]"),
        ("solve((-2/3)*x+3/7=1/2,x)", "Result: [{x: -3/28}]"),
        ("solve((x-1)*(x-5)*(x+5)=0,x)", "Result: [{x: -5}, {x: 1}, {x: 5}]"),
        ("solve(6/x+8/(2*x)=10,x)", "Result: [{x: 1}]"),
        ("solve((3+x)/(x^2+3)=1,x)", "Result: [{x: 0}, {x: 1}]"),
    ],
)
def test_adapted_equation_cases_keep_complete_classroom_derivations(
    solver: Solver,
    query: str,
    expected_result: str,
) -> None:
    """Hard linear, factored, and rational equations should expose every main move."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert len(result.steps) >= _MIN_EQUATION_STEPS
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert all(step.before != step.after for step in result.steps)
    assert format_ascii(result).endswith(expected_result)


@pytest.mark.parametrize(
    ("query", "expected_rule", "expected_result"),
    [
        ("factor(x^2-4)", "Factor the expression", "Result: (x - 2) * (x + 2)"),
        ("expand((x*y)^2)", "Expand the expression", "Result: x ^ 2 * y ^ 2"),
        (
            "apart((x+1)/(x*(x-1)),x)",
            "Decompose into partial fractions",
            "Result: 2 / (x - 1) - 1 / x",
        ),
        (
            "simplify(sin(x)^2+cos(x)^2)",
            "Apply algebraic identities",
            "Result: 1",
        ),
    ],
)
def test_other_algebra_operations_also_avoid_opaque_backend_steps(
    solver: Solver,
    query: str,
    expected_rule: str,
    expected_result: str,
) -> None:
    """Every elementary algebra entry point should explain its mathematical action."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (expected_rule,)
    assert format_ascii(result).endswith(expected_result)


def test_explicit_cancel_operation_states_the_nonzero_factor(solver: Solver) -> None:
    """The cancel operation should make the lost input-domain value visible."""
    result = solver.solve("cancel((x^2-1)/(x-1))")
    assert isinstance(result, ExactResult)
    assert result.steps[0].rule == "Cancel common factors"
    assert tuple(
        format_expression(constraint.expression)
        for constraint in result.steps[0].introduced_constraints
    ) == ("x - 1 != 0",)
    assert format_ascii(result).endswith("Result: x + 1")
