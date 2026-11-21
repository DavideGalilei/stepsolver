"""Finite and infinite limits solved by rationalization."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendLimit,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_POWER_ARITY = 2


def derive_infinite_radical_conjugate(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Rationalize ``x*(sqrt(x**2+c)-x)`` at positive infinity."""
    if point != sp.oo:
        return ()
    difference = sp.simplify(expression / variable)
    radicand = sp.simplify((difference + variable) ** 2)
    constant = sp.simplify(radicand - variable**2)
    if constant.has(variable) or constant.is_positive is not True:
        return ()
    root = sp.sqrt(variable**2 + constant)
    if sp.simplify(difference - (root - variable)) != sp.Integer(0):
        return ()
    rationalized = constant * variable / (root + variable)
    normalized = constant / (sp.sqrt(1 + constant / variable**2) + 1)
    if sp.simplify(result - constant / 2) != sp.Integer(0):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    rationalized_limit = BackendLimit(
        expression=rationalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    normalized_limit = BackendLimit(
        expression=normalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Multiply by the conjugate",
            before=displayed,
            after=rationalized_limit,
            explanation="Use the conjugate so the difference of squares removes the subtraction.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Multiplying by the conjugate quotient preserves the expression.",
            notes=(BackendMathNote(label="Conjugate", expression=root + variable),),
        ),
        BackendDerivationStep(
            rule="Divide by the leading variable",
            before=rationalized_limit,
            after=normalized_limit,
            explanation=(
                "Divide numerator and denominator by the positive variable to expose terms "
                "that vanish at infinity."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The normalized quotient is equal for sufficiently large values.",
        ),
        BackendDerivationStep(
            rule="Evaluate the normalized limit",
            before=normalized_limit,
            after=result,
            explanation="The reciprocal-square term tends to zero, so substitute its limit.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Continuity of the square root gives the exact value.",
        ),
    )


def _square_root_terms(expression: sp.Basic) -> tuple[sp.Basic, ...]:
    current = (
        (expression,)
        if expression.is_Pow
        and len(expression.args) == _POWER_ARITY
        and expression.args[1] == sp.Rational(1, 2)
        else ()
    )
    nested = tuple(item for argument in expression.args for item in _square_root_terms(argument))
    return (*current, *nested)


def derive_radical_rationalization(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Rationalize a square-root difference that produces zero over zero."""
    rationalization = _radical_rationalization(expression, variable)
    if point in {sp.oo, -sp.oo} or rationalization is None:
        return ()
    conjugate, rationalized = rationalization
    substituted = sp.simplify(rationalized.subs(variable, point))
    if substituted != result or substituted.has(sp.zoo) or substituted.has(sp.nan):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    rationalized_limit = BackendLimit(
        expression=rationalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Multiply by the conjugate",
            before=displayed,
            after=rationalized_limit,
            explanation=(
                "Multiply numerator and denominator by the conjugate, then use the difference "
                "of squares to remove the radical from the numerator."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The conjugate quotient equals one away from the hole.",
            notes=(BackendMathNote(label="Conjugate", expression=conjugate),),
        ),
        BackendDerivationStep(
            rule="Substitute into the rationalized expression",
            before=rationalized_limit,
            after=result,
            explanation="The rationalized expression is continuous at the approach point.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Direct substitution gives the exact limit.",
        ),
    )


def _radical_rationalization(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic] | None:
    numerator, denominator = sp.fraction(sp.together(expression))
    square_roots = _square_root_terms(numerator)
    if len(square_roots) != 1:
        return None
    root = square_roots[0]
    constant = sp.simplify(root - numerator)
    if constant.has(variable) or constant == sp.Integer(0):
        return None
    if sp.simplify(numerator - (root - constant)) != sp.Integer(0):
        return None
    conjugate = root + constant
    radicand = root.args[0]
    rationalized = sp.cancel((radicand - constant**2) / (denominator * conjugate))
    if sp.simplify(rationalized - expression) != sp.Integer(0):
        return None
    return conjugate, rationalized
