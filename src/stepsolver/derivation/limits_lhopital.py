"""Human-readable L'Hopital derivations for finite indeterminate limits."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendEvaluationAtIndex,
    BackendIdentity,
    BackendInlineMath,
    BackendLimit,
    BackendMathNote,
    BackendQuotient,
)
from stepsolver.results import VerificationMethod

_MAX_LHOPITAL_ROUNDS = 3
_LHOPITAL_FUNCTIONS = (
    sp.exp,
    sp.log,
    sp.sin,
    sp.cos,
    sp.tan,
    sp.sinh,
    sp.cosh,
    sp.tanh,
)


def _substitution_value(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
) -> sp.Basic | None:
    """Evaluate one side of a quotient by direct finite substitution."""
    try:
        value = sp.simplify(expression.subs(variable, point))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if value not in {sp.nan, sp.zoo} else None


def _is_zero_over_zero(
    numerator: sp.Basic,
    denominator: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
) -> bool:
    """Return whether direct substitution produces the indeterminate form zero over zero."""
    return _substitution_value(numerator, variable, point) == sp.Integer(0) and _substitution_value(
        denominator, variable, point
    ) == sp.Integer(0)


def derive_lhopital_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Resolve a finite zero-over-zero limit with explicit L'Hopital rounds."""
    if point in {sp.oo, -sp.oo} or getattr(result, "is_finite", None) is not True:
        return ()
    numerator, denominator = sp.fraction(sp.together(expression))
    if (
        denominator == sp.Integer(1)
        or not any(
            numerator.has(function) or denominator.has(function) for function in _LHOPITAL_FUNCTIONS
        )
        or not _is_zero_over_zero(
            numerator,
            denominator,
            variable,
            point,
        )
    ):
        return ()

    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    point_identity = BackendIdentity(left=variable, right=point)
    zero_over_zero = BackendQuotient(
        numerator=sp.Integer(0),
        denominator=sp.Integer(0),
    )
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Check direct substitution",
            before=displayed_limit,
            after=zero_over_zero,
            explanation=(
                f"Substitute {variable} = {point}. Both the numerator and denominator become "
                "zero, so the limit has the indeterminate form 0/0."
            ),
            explanation_parts=(
                "Substitute ",
                BackendInlineMath(expression=point_identity),
                (
                    ". Both the numerator and denominator become zero, giving the "
                    "indeterminate form "
                ),
                BackendInlineMath(expression=zero_over_zero),
                ".",
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Direct substitution gives zero in both parts of the quotient.",
            notes=(
                BackendMathNote(
                    label="Numerator",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtIndex(
                            expression=numerator,
                            variable=variable,
                            index=point,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
                BackendMathNote(
                    label="Denominator",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtIndex(
                            expression=denominator,
                            variable=variable,
                            index=point,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
            ),
        ),
    ]

    current_numerator = numerator
    current_denominator = denominator
    current_limit = displayed_limit
    for round_number in range(1, _MAX_LHOPITAL_ROUNDS + 1):
        differentiated_numerator = sp.diff(current_numerator, variable)
        differentiated_denominator = sp.diff(current_denominator, variable)
        if differentiated_denominator == sp.Integer(0):
            return ()
        differentiated_quotient = BackendQuotient(
            numerator=differentiated_numerator,
            denominator=differentiated_denominator,
        )
        differentiated_limit = BackendLimit(
            expression=differentiated_quotient,
            variable=variable,
            point=point,
            direction=direction,
        )
        match round_number:
            case 1:
                rule = "Apply L'Hôpital's rule"
                explanation = (
                    "Differentiate the numerator and denominator separately, then form their "
                    "new quotient inside the limit."
                )
            case _:
                rule = "Apply L'Hôpital's rule again"
                explanation = (
                    "Direct substitution is still 0/0, so differentiate the new numerator and "
                    "denominator once more."
                )
        steps.append(
            BackendDerivationStep(
                rule=rule,
                before=current_limit,
                after=differentiated_limit,
                explanation=explanation,
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "L'Hôpital's rule preserves the limit because the quotient has the "
                    "indeterminate form 0/0."
                ),
                notes=(
                    BackendMathNote(
                        label="Differentiate the numerator",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=current_numerator,
                                variable=variable,
                            ),
                            right=differentiated_numerator,
                        ),
                    ),
                    BackendMathNote(
                        label="Differentiate the denominator",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=current_denominator,
                                variable=variable,
                            ),
                            right=differentiated_denominator,
                        ),
                    ),
                ),
            ),
        )
        current_numerator = differentiated_numerator
        current_denominator = differentiated_denominator
        current_limit = differentiated_limit
        if not _is_zero_over_zero(
            current_numerator,
            current_denominator,
            variable,
            point,
        ):
            break
    else:
        return ()

    numerator_value = _substitution_value(current_numerator, variable, point)
    denominator_value = _substitution_value(current_denominator, variable, point)
    if (
        numerator_value is None
        or denominator_value is None
        or denominator_value == sp.Integer(0)
        or sp.simplify(numerator_value / denominator_value - result) != sp.Integer(0)
    ):
        return ()
    substituted_quotient = BackendQuotient(
        numerator=numerator_value,
        denominator=denominator_value,
    )
    steps.append(
        BackendDerivationStep(
            rule="Substitute into the transformed limit",
            before=current_limit,
            after=result,
            explanation=(
                f"The transformed quotient is now defined at {variable} = {point}, so substitute "
                "the approach value and simplify."
            ),
            explanation_parts=(
                "The transformed quotient is now defined at ",
                BackendInlineMath(expression=point_identity),
                ", so direct substitution gives ",
                BackendInlineMath(
                    expression=BackendIdentity(
                        left=substituted_quotient,
                        right=result,
                    ),
                ),
                ".",
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Direct substitution in the differentiated quotient is defined.",
            notes=(
                BackendMathNote(
                    label="Standard identity check",
                    expression=BackendIdentity(left=displayed_limit, right=result),
                ),
            ),
        ),
    )
    return tuple(steps)
