"""Finite limits by cancellation, sign, or substitution."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
)
from stepsolver.results import VerificationMethod


def derive_finite_algebraic_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use cancellation, one-sided sign analysis, or substitution at a finite point."""
    if point in {sp.oo, -sp.oo}:
        return ()
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )

    numerator, denominator = sp.fraction(sp.together(expression))
    numerator_at_point = sp.simplify(numerator.subs(variable, point))
    denominator_at_point = sp.simplify(denominator.subs(variable, point))
    if numerator_at_point == sp.Integer(0) and denominator_at_point == sp.Integer(0):
        canceled = sp.cancel(expression)
        if str(canceled) != str(expression):
            canceled_limit = BackendLimit(
                expression=canceled,
                variable=variable,
                point=point,
                direction=direction,
            )
            return (
                BackendDerivationStep(
                    rule="Factor and cancel the common factor",
                    before=displayed_limit,
                    after=canceled_limit,
                    explanation=(
                        "Factor the numerator and denominator, then cancel the common "
                        "factor for nearby values."
                    ),
                    verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    verification_detail=(
                        "The original and canceled expressions agree away from the hole."
                    ),
                    notes=(
                        BackendMathNote(
                            label="The limit ignores the value at the hole",
                            expression=BackendNotEqual(left=variable, right=point),
                        ),
                    ),
                ),
                BackendDerivationStep(
                    rule="Substitute into the simplified expression",
                    before=canceled_limit,
                    after=result,
                    explanation=(
                        "The simplified expression is continuous at the approach point, "
                        "so substitute directly."
                    ),
                    verification_method=VerificationMethod.EXACT_ARITHMETIC,
                    verification_detail="Direct substitution gives the exact limit.",
                ),
            )

    if direction in {"+", "-"} and denominator == variable - point:
        expected = sp.oo if direction == "+" else -sp.oo
        if result == expected and numerator == sp.Integer(1):
            side_symbol = sp.Symbol("u", positive=True)
            signed_denominator = side_symbol if direction == "+" else -side_symbol
            return (
                BackendDerivationStep(
                    rule="Analyze the sign from the requested side",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "The denominator approaches zero through positive values."
                        if direction == "+"
                        else "The denominator approaches zero through negative values."
                    ),
                    verification_method=VerificationMethod.EXACT_ARITHMETIC,
                    verification_detail=(
                        "The reciprocal grows without bound with the indicated sign."
                    ),
                    notes=(
                        BackendMathNote(
                            label="Nearby denominator sign",
                            expression=BackendIdentity(
                                left=denominator,
                                right=signed_denominator,
                            ),
                        ),
                    ),
                ),
            )

    substituted = sp.simplify(expression.subs(variable, point))
    if substituted == result and not substituted.has(sp.zoo) and not substituted.has(sp.nan):
        return (
            BackendDerivationStep(
                rule="Use direct substitution",
                before=displayed_limit,
                after=result,
                explanation=(
                    "The expression is continuous at the approach point, so evaluate it "
                    "there directly."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="Exact substitution gives the displayed value.",
            ),
        )

    return ()
