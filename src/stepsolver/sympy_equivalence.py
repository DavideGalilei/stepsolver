"""Tiered exact-equivalence checks for symbolic expressions."""

from __future__ import annotations

import sympy as sp


def _polynomial_decision(difference: sp.Basic) -> bool | None:
    expanded = sp.expand(difference)
    if expanded == sp.Integer(0):
        return True
    return None if expanded.has(sp.Float) else False


def _rational_decision(difference: sp.Basic) -> bool | None:
    numerator, _denominator = sp.fraction(sp.together(difference))
    expanded_numerator = sp.expand(numerator)
    if expanded_numerator == sp.Integer(0):
        return True
    return None if expanded_numerator.has(sp.Float) else False


def _trigonometric_decision(difference: sp.Basic) -> bool | None:
    if sp.trigsimp(difference) == sp.Integer(0):
        return True
    return None


def fast_equivalence(
    left: sp.Basic,
    right: sp.Basic,
    *,
    variable: sp.Symbol | None = None,
) -> bool | None:
    """Prove equality or inequality cheaply, returning ``None`` when inconclusive.

    The fast tiers are deliberately narrow.  They handle exact polynomial and
    rational differences, plus a small trigonometric-identity tier.  Anything
    outside those classes falls back to the caller's stronger verifier so that
    branch cuts, assumptions, and non-algebraic identities are not guessed.
    """
    difference = left - right
    if difference == sp.Integer(0):
        return True
    if difference.is_number:
        if difference.has(sp.Float):
            return None
        return sp.simplify(difference) == sp.Integer(0)
    variables = difference.free_symbols
    if not variables or (variable is not None and not variables.issubset({variable})):
        return None
    decision: bool | None = None
    if difference.is_polynomial(*variables):
        decision = _polynomial_decision(difference)
    elif difference.is_rational_function(*variables):
        decision = _rational_decision(difference)
    elif difference.has(sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh):
        decision = _trigonometric_decision(difference)
    return decision


def expressions_equivalent(
    left: sp.Basic,
    right: sp.Basic,
    *,
    variable: sp.Symbol | None = None,
) -> bool:
    """Return whether two expressions are exactly equivalent on their domain.

    A conclusive fast result is returned immediately.  Inconclusive structural
    checks continue to SymPy's full simplifier rather than treating uncertainty
    as equality.
    """
    fast = fast_equivalence(left, right, variable=variable)
    if fast is not None:
        return fast
    return sp.simplify(left - right) == sp.Integer(0)
