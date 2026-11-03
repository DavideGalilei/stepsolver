"""Domain restrictions and exact rational-root candidate checks."""

import pytest
import sympy as sp

from stepsolver import ExactResult, Solver, format_ascii
from stepsolver.derivation.equations_polynomial import derive_cubic_steps
from stepsolver.derivation.equations_support import evaluate_polynomial_at_rational


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


def test_l01_denominator_exclusion_is_preserved(solver: Solver) -> None:
    """The cancelled denominator must not reappear as an extraneous solution."""
    result = solver.solve("solve((x^2-1)/(x-1)=0,x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert "x != 1" in rendered
    assert rendered.rstrip().endswith("Result: [{x: -1}]")


def test_l02_squaring_does_not_return_the_extraneous_root(solver: Solver) -> None:
    """The original square-root equation keeps only its valid real solution."""
    result = solver.solve("solve(sqrt(x+1)=x-1,x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert rendered.rstrip().endswith("Result: [{x: 3}]")
    assert "Compute exact result" not in rendered
    rules = tuple(step.rule for step in result.steps)
    assert "Square both sides" in rules
    assert "Check candidates in the original equation" in rules
    assert any(note.label == "Test x = 0 (extraneous)" for note in result.steps[-1].notes)


def test_reversed_square_root_equation_has_the_same_valid_root(solver: Solver) -> None:
    """The orientation of the radical should not change the candidate filter."""
    result = solver.solve("solve(x-1=sqrt(x+1),x)")
    assert isinstance(result, ExactResult)
    assert format_ascii(result).rstrip().endswith("Result: [{x: 3}]")


def test_square_root_equation_with_no_valid_candidate_is_not_opaque(
    solver: Solver,
) -> None:
    """Squaring can produce only extraneous candidates; that case needs a derivation."""
    result = solver.solve("solve(sqrt(x+1)=-1,x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert rendered.rstrip().endswith("Result: No solution")
    assert "Compute exact result" not in rendered
    assert "Check candidates in the original equation" in rendered


def test_rational_root_candidate_uses_exact_horner_evaluation() -> None:
    """Candidate checks should be exact fractions rather than floating estimates."""
    x = sp.Symbol("x", real=True)
    polynomial = sp.Poly(2 * x**3 - 3 * x**2 + 5 * x - 7, x)
    assert evaluate_polynomial_at_rational(
        polynomial,
        sp.Rational(1, 2),
    ) == sp.Integer(-5)
    assert evaluate_polynomial_at_rational(polynomial, sp.Integer(1)) == sp.Integer(-3)


def test_cubic_without_a_rational_root_skips_factoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An irreducible rational cubic should go straight to its numerical method."""
    x = sp.Symbol("x", real=True)
    polynomial = sp.Poly(x**3 + x + 1, x)
    roots = (sp.CRootOf(x**3 + x + 1, 0),)

    def fail_factor(*_args: object, **_kwargs: object) -> object:
        message = "factoring should be skipped when no rational candidate is a root"
        raise AssertionError(message)

    monkeypatch.setattr(sp, "factor", fail_factor)
    steps = derive_cubic_steps(
        sp.Eq(polynomial.as_expr(), 0),
        x,
        polynomial,
        roots,
        (),
    )
    assert steps
    assert steps[0].rule == "Approximate the real root"
