"""Tiered equivalence checks and their trusted fallback."""

import sympy as sp

from stepsolver.sympy_equivalence import expressions_equivalent, fast_equivalence


def test_polynomial_identity_and_non_identity_are_decided_quickly() -> None:
    """Exact polynomial residuals should be decided without the full simplifier."""
    x = sp.Symbol("x", real=True)
    assert fast_equivalence(sp.expand((x + 1) ** 2), x**2 + 2 * x + 1) is True
    assert fast_equivalence(x**2, x**3) is False


def test_rational_identity_is_proven_with_exact_numerator_expansion() -> None:
    """Cancellation is an algebraic identity on the common domain."""
    x = sp.Symbol("x", real=True)
    assert expressions_equivalent((x**2 - 1) / (x - 1), x + 1) is True
    assert fast_equivalence((x**2 - 1) / (x - 1), x + 1) is True


def test_trigonometric_identity_uses_the_specialized_tier() -> None:
    """The Pythagorean identity should not require the general simplifier."""
    x = sp.Symbol("x", real=True)
    assert fast_equivalence(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)) is True


def test_logarithmic_identity_continues_to_the_trusted_fallback() -> None:
    """Logarithmic identities may be inconclusive at the fast tiers."""
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y", positive=True)
    assert fast_equivalence(sp.log(x * y), sp.log(x) + sp.log(y)) is None
    assert expressions_equivalent(sp.log(x * y), sp.log(x) + sp.log(y)) is True


def test_inconclusive_expression_is_not_treated_as_equal() -> None:
    """An unknown identity must return None, never a guessed True."""
    x = sp.Symbol("x", real=True)
    assert fast_equivalence(sp.sin(x), sp.cos(x)) is None
    assert expressions_equivalent(sp.sin(x), sp.cos(x)) is False


def test_fast_decisions_agree_with_the_trusted_simplifier_on_a_corpus() -> None:
    """Every conclusive fast decision must agree with the old oracle."""
    x = sp.Symbol("x", real=True)
    corpus = (
        (sp.Integer(2), sp.Integer(2)),
        (sp.Integer(2), sp.Integer(3)),
        (x**2 - 1, (x - 1) * (x + 1)),
        (x**2 + 1, (x - 1) * (x + 1)),
        ((x**2 - 1) / (x - 1), x + 1),
        (1 / (x - 1), 1 / (x - 1)),
        (sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)),
        (sp.sin(x), sp.cos(x)),
        (sp.log(sp.exp(x)), x),
        (sp.exp(sp.log(x)), x),
        (sp.Abs(x), x),
    )
    for left, right in corpus:
        decision = fast_equivalence(left, right, variable=x)
        if decision is not None:
            assert decision == (sp.simplify(left - right) == sp.Integer(0))
