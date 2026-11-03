"""Squeeze-theorem derivations for bounded oscillating factors."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
    BackendRelation,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod

_MIN_PRODUCT_FACTORS = 2


def derive_bounded_oscillation_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Squeeze a vanishing factor times a bounded sine or cosine."""
    if result != sp.Integer(0):
        return ()
    factors = expression.as_ordered_factors()
    if len(factors) < _MIN_PRODUCT_FACTORS:
        return ()
    bounded: sp.Basic | None = None
    vanishing: sp.Basic | None = None
    for index, factor in enumerate(factors):
        if type(factor).__name__ not in {"sin", "cos"} or len(factor.args) != 1:
            continue
        argument = factor.args[0]
        scaled = sp.simplify(argument * (variable - point))
        if scaled.has(variable) or scaled == sp.Integer(0):
            continue
        bounded = factor
        vanishing = sp.Mul(*(item for position, item in enumerate(factors) if position != index))
        break
    if bounded is None or vanishing is None:
        return ()
    try:
        vanishing_limit = sp.limit(vanishing, variable, point)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if sp.simplify(vanishing_limit) != sp.Integer(0):
        return ()
    bound = sp.Abs(vanishing)
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    bound_limit = BackendLimit(
        expression=bound,
        variable=variable,
        point=point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Bound the oscillating factor",
            before=displayed,
            after=(
                BackendRelation(operator="<=", left=sp.Integer(-1), right=bounded),
                BackendRelation(operator="<=", left=bounded, right=sp.Integer(1)),
            ),
            explanation=(
                "A sine or cosine takes every value between -1 and 1, so the oscillating "
                "factor stays inside that interval wherever it is defined."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The bounded factor is a sine or cosine, whose real range is [-1, 1]."
            ),
            introduced_constraints=(
                BackendStepConstraint(
                    explanation="The oscillating factor requires a nonzero argument.",
                    expression=BackendNotEqual(left=variable, right=point),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Multiply the bounds by the vanishing factor",
            before=(
                BackendRelation(operator="<=", left=sp.Integer(-1), right=bounded),
                BackendRelation(operator="<=", left=bounded, right=sp.Integer(1)),
            ),
            after=(
                BackendRelation(operator="<=", left=-bound, right=expression),
                BackendRelation(operator="<=", left=expression, right=bound),
            ),
            explanation=(
                "Multiply the oscillation bounds by the remaining factor. Taking the "
                "absolute value of that factor gives one lower and one upper bound that "
                "hold whichever sign the factor has."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "Multiplying -1 <= f <= 1 by g yields -|g| <= f*g <= |g| for either sign of g."
            ),
        ),
        BackendDerivationStep(
            rule="Apply the squeeze theorem",
            before=BackendLimit(
                expression=bound,
                variable=variable,
                point=point,
                direction=direction,
            ),
            after=result,
            explanation=(
                "Both bounding expressions tend to 0, so the squeezed product has the same "
                "limit by the squeeze theorem."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The lower and upper bounding limits are both zero, so the squeezed "
                "expression tends to zero."
            ),
            notes=(
                BackendMathNote(
                    label="Lower product bound",
                    expression=BackendRelation(operator="<=", left=-bound, right=expression),
                ),
                BackendMathNote(
                    label="Upper product bound",
                    expression=BackendRelation(operator="<=", left=expression, right=bound),
                ),
                BackendMathNote(
                    label="Bounding limit",
                    expression=BackendIdentity(left=bound_limit, right=sp.Integer(0)),
                ),
            ),
        ),
    )
