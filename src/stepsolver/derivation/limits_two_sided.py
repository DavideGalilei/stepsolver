"""Two-sided limits that fail because the one-sided limits disagree."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendUndefined,
)
from stepsolver.results import VerificationMethod

_ACCUM_BOUNDS = getattr(sp, "AccumBounds")


def derive_two_sided_nonexistence(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Show that a two-sided limit fails because the one-sided limits disagree."""
    if (direction is not None or result not in {sp.nan, sp.zoo}) and not result.has(_ACCUM_BOUNDS):
        return ()
    if point.has(sp.oo, -sp.oo):
        return ()
    try:
        left = sp.limit(expression, variable, point, "-")
        right = sp.limit(expression, variable, point, "+")
    except (NotImplementedError, TypeError, ValueError):
        return ()
    oscillates = left.has(_ACCUM_BOUNDS) or right.has(_ACCUM_BOUNDS)
    if not oscillates and left == right and left not in {sp.nan, sp.zoo}:
        return ()
    displayed = BackendLimit(expression=expression, variable=variable, point=point)
    if oscillates:
        return (
            BackendDerivationStep(
                rule="Examine the one-sided behaviour",
                before=displayed,
                after=BackendUndefined(),
                explanation=(
                    "The expression oscillates without settling on a single value as the "
                    "variable approaches the point, so the two-sided limit does not exist."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail=(
                    "Neither one-sided approach settles on a single finite value, so no "
                    "two-sided limit exists."
                ),
            ),
        )
    left_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction="-",
    )
    right_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction="+",
    )
    return (
        BackendDerivationStep(
            rule="Evaluate the left-hand limit",
            before=left_limit,
            after=left,
            explanation=(
                "Approach the point through smaller values. This one-sided limit is computed "
                "independently of the other side."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The left-hand limit is the displayed exact value.",
        ),
        BackendDerivationStep(
            rule="Evaluate the right-hand limit",
            before=right_limit,
            after=right,
            explanation=(
                "Approach the point through larger values. Both finite points must give the "
                "same value before a two-sided limit can exist."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The right-hand limit is the displayed exact value.",
        ),
        BackendDerivationStep(
            rule="Conclude the two-sided limit does not exist",
            before=displayed,
            after=BackendUndefined(),
            explanation=(
                "The left-hand and right-hand limits are different, so the two-sided limit "
                "does not exist."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=("A two-sided limit exists only when both one-sided limits agree."),
            notes=(
                BackendMathNote(
                    label="Left-hand limit",
                    expression=BackendIdentity(left=left_limit, right=left),
                ),
                BackendMathNote(
                    label="Right-hand limit",
                    expression=BackendIdentity(left=right_limit, right=right),
                ),
            ),
        ),
    )
