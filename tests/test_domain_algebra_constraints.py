"""Domain semantics, rational constraints, and radical-normalization regressions."""

from dataclasses import replace

import pytest

from stepsolver import (
    ExactResult,
    NoSolutionValue,
    QueryError,
    ScalarValue,
    SequenceValue,
    SolutionDomain,
    Solver,
    format_ascii,
    format_expression,
    parse,
)


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


def _constraints(result: ExactResult) -> tuple[str, ...]:
    """Render every domain restriction introduced anywhere in a derivation."""
    return tuple(
        format_expression(item.expression)
        for step in result.steps
        for item in step.introduced_constraints
    )


def test_real_domain_discards_nonreal_quadratic_roots(solver: Solver) -> None:
    """A real-domain quadratic with a negative discriminant has no solutions."""
    result = solver.solve("solve(x^2 + 4 = 0, x)", domain="real")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Calculate the discriminant",
        "Conclude there are no real solutions",
    )
    assert "Compute exact result" not in rules


def test_complex_domain_keeps_both_imaginary_roots(solver: Solver) -> None:
    """An explicit complex domain keeps the non-real roots and their derivation."""
    result = solver.solve("solve(x^2 + 4 = 0, x)", domain="complex")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, SequenceValue)
    rendered = format_ascii(result)
    assert "Result: [{x: -2 * i}, {x: 2 * i}]" in rendered
    assert result.steps


def test_omitted_domain_preserves_complex_polynomial_default(solver: Solver) -> None:
    """The documented default continues to solve isolated polynomials over C."""
    result = solver.solve("solve(x^2 + 4 = 0, x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, SequenceValue)
    assert "Result: [{x: -2 * i}, {x: 2 * i}]" in format_ascii(result)


def test_real_domain_cubic_keeps_only_real_roots(solver: Solver) -> None:
    """A real-domain cubic must omit the non-real factor with an explicit reason."""
    result = solver.solve("solve(x^3 - x^2 + x - 1 = 0, x)", domain="real")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, SequenceValue)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Factor the polynomial",
        "Apply the zero-product property",
        "Keep the real solutions",
    )
    rendered = format_ascii(result)
    assert "No real zeros: x ^ 2 + 1 = 0" in rendered
    assert rendered.endswith("Result: [{x: 1}]")


def test_real_domain_quartic_without_real_roots_has_human_steps(solver: Solver) -> None:
    """A real-domain quartic with only complex roots must not fall back opaquely."""
    result = solver.solve("solve(x^4 + 1 = 0, x)", domain="real")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert tuple(step.rule for step in result.steps) == ("Conclude there are no real solutions",)
    assert result.steps[0].verification.method.value == "exact arithmetic"


@pytest.mark.parametrize("domain", ["integer", "", "Real numbers"])
def test_invalid_domain_is_rejected(solver: Solver, domain: str) -> None:
    """Only the two documented ambient domains are accepted."""
    with pytest.raises(QueryError, match="domain must be 'real' or 'complex'"):
        solver.solve("solve(x^2 + 4 = 0, x)", domain=domain)


def test_domain_is_recorded_on_a_parsed_query(solver: Solver) -> None:
    """An already parsed query can carry its domain without a second parse."""
    query = parse("solve(x^2 + 4 = 0, x)")
    result = solver.solve(query, domain=SolutionDomain.REAL)
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)


def test_conflicting_query_and_call_domains_are_rejected(solver: Solver) -> None:
    """A caller cannot silently override a domain already recorded on the query."""
    query = replace(
        parse("solve(x^2 + 4 = 0, x)"),
        domain=SolutionDomain.COMPLEX,
    )
    with pytest.raises(QueryError, match="conflicts with the domain"):
        solver.solve(query, domain="real")


def test_simplification_keeps_both_original_exclusions(solver: Solver) -> None:
    """Cancelling x - 2 must not erase either original denominator zero."""
    result = solver.solve("simplify((x^3 - 8)/(x^2 - 4))")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, ScalarValue)
    assert format_expression(result.value.expression) == "(x ^ 2 + 2 * x + 4) / (x + 2)"
    assert result.steps[0].rule == "Cancel common factors"
    assert set(_constraints(result)) == {"x != -2", "x != 2"}
    assert "wherever the original denominators are nonzero" in (result.steps[0].verification.detail)


def test_nested_numeric_radical_is_denested_with_derivation(solver: Solver) -> None:
    """A numeric square root of a perfect square is rewritten with a positive-root check."""
    result = solver.solve("sqrt(7 + 4*sqrt(3))")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, ScalarValue)
    assert format_expression(result.value.expression) == "sqrt(3) + 2"
    assert tuple(step.rule for step in result.steps) == (
        "Rewrite the radicand as a perfect square",
        "Take the positive square root",
    )
    rendered = format_ascii(result)
    assert "Perfect square: 4 * sqrt(3) + 7 = (sqrt(3) + 2) ^ 2" in rendered
    assert "Positive root: sqrt(3) + 2 > 0" in rendered


def test_partial_fractions_report_original_denominator_roots(solver: Solver) -> None:
    """The decomposition is valid only where both original factors are nonzero."""
    result = solver.solve("apart(1/(x^2*(x+1)),x)")
    assert isinstance(result, ExactResult)
    assert result.steps[0].rule == "Decompose into partial fractions"
    assert set(_constraints(result)) == {"x != 0", "x != -1"}
    assert "x ^ (-2)" in format_ascii(result)


def test_repeated_denominator_roots_are_reported_once(solver: Solver) -> None:
    """A repeated denominator factor contributes a single exclusion."""
    result = solver.solve("simplify((x^3)/(x^2))")
    assert isinstance(result, ExactResult)
    assert _constraints(result) == ("x != 0",)


def test_denominator_without_real_zeros_adds_no_real_exclusion(solver: Solver) -> None:
    """An irreducible real quadratic denominator never vanishes over the reals."""
    result = solver.solve("simplify((x^3 + x)/(x^2 + 1))")
    assert isinstance(result, ExactResult)
    assert _constraints(result) == ()
    assert result.steps[0].rule == "Cancel common factors"


def test_quartic_denominator_without_real_zeros_adds_no_exclusion(solver: Solver) -> None:
    """A denominator whose roots are all non-real contributes no real exclusion."""
    result = solver.solve("simplify((x^2 + 1)/(x^4 + 1))")
    assert isinstance(result, ExactResult)
    assert _constraints(result) == ()


def test_multivariate_denominator_keeps_symbolic_nonzero_constraint(solver: Solver) -> None:
    """When no single-variable real-root set applies, keep the symbolic restriction."""
    result = solver.solve("simplify((x*(x^2 + y))/(x^2 + y))")
    assert isinstance(result, ExactResult)
    assert _constraints(result) == ("x ^ 2 + y != 0",)


def test_non_denestable_nested_radical_keeps_its_radical_form(solver: Solver) -> None:
    """A nested radical that is not a perfect square must not be rewritten."""
    result = solver.solve("sqrt(2 + sqrt(2))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Apply algebraic identities",)
    assert "Rewrite the radicand as a perfect square" not in format_ascii(result)


def test_non_square_power_is_not_treated_as_a_radical(solver: Solver) -> None:
    """A cube root passes through the ordinary simplification path."""
    result = solver.solve("simplify(2^(1/3))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Evaluate the arithmetic",)
    assert format_ascii(result).endswith("Result: 2 ^ (1 / 3)")


def test_irrational_square_root_remains_unchanged(solver: Solver) -> None:
    """A non-square integer radicand has no exact denesting."""
    result = solver.solve("simplify(sqrt(5))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Apply algebraic identities",)
    assert format_ascii(result).endswith("Result: sqrt(5)")


def test_real_quadratic_is_standardized_before_applying_the_formula(solver: Solver) -> None:
    """A quadratic with a nonzero right-hand side is moved to standard form first."""
    result = solver.solve("solve(2*x^2 - 3*x + 1 = 5, x)", domain="real")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, SequenceValue)
    assert tuple(step.rule for step in result.steps) == (
        "Write in standard form",
        "Apply the quadratic formula",
    )
    rendered = format_ascii(result)
    assert "Eq(2 * x ^ 2 - 3 * x - 4, 0)" in rendered
    assert "Discriminant: delta = 41" in rendered
