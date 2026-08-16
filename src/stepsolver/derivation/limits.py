"""Human-readable finite and infinite limit strategies."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendQuotient,
)
from stepsolver.results import VerificationMethod


def _derive_infinite_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    numerator, denominator = sp.fraction(sp.together(expression))
    try:
        numerator_polynomial = sp.Poly(numerator, variable)
        denominator_polynomial = sp.Poly(denominator, variable)
    except sp.PolynomialError:
        numerator_polynomial = None
        denominator_polynomial = None
    if numerator_polynomial is not None and denominator_polynomial is not None:
        numerator_degree = numerator_polynomial.degree()
        denominator_degree = denominator_polynomial.degree()
        if numerator_degree <= denominator_degree:
            numerator_leading = (
                numerator_polynomial.coeff_monomial(variable**numerator_degree)
                * variable**numerator_degree
            )
            denominator_leading = (
                denominator_polynomial.coeff_monomial(variable**denominator_degree)
                * variable**denominator_degree
            )
            return (
                BackendDerivationStep(
                    rule="Compare the leading powers",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "At infinity, the highest-degree terms determine the ratio's limiting "
                        "behavior."
                    ),
                    verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    verification_detail=(
                        "Lower-degree terms vanish relative to the leading powers."
                    ),
                    notes=(
                        BackendMathNote(
                            label="Leading-term ratio",
                            expression=BackendLimit(
                                expression=numerator_leading / denominator_leading,
                                variable=variable,
                                point=point,
                            ),
                        ),
                    ),
                ),
            )
    if expression.has(sp.exp(variable)) and result in {sp.oo, -sp.oo}:
        generic_exponent = sp.Symbol("n", positive=True)
        return (
            BackendDerivationStep(
                rule="Use exponential growth",
                before=displayed_limit,
                after=result,
                explanation="An exponential grows faster than every fixed power of the variable.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The standard exponential-over-power growth rule applies.",
                notes=(
                    BackendMathNote(
                        label="Growth rule",
                        expression=BackendIdentity(
                            left=BackendLimit(
                                expression=sp.exp(variable) / variable**generic_exponent,
                                variable=variable,
                                point=sp.oo,
                            ),
                            right=sp.oo,
                        ),
                    ),
                ),
            ),
        )
    return ()


def _derive_sine_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    standard_sine_quotient = sp.sin(variable) / variable
    if point == sp.Integer(0) and sp.simplify(expression - standard_sine_quotient) == sp.Integer(0):
        generic_variable = sp.Symbol("u", real=True)
        return (
            BackendDerivationStep(
                rule="Use the standard sine limit",
                before=displayed_limit,
                after=result,
                explanation=("This expression is the standard trigonometric limit at zero."),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The expression exactly matches the standard sine limit.",
                notes=(
                    BackendMathNote(
                        label="Standard limit",
                        expression=BackendIdentity(
                            left=BackendLimit(
                                expression=sp.sin(generic_variable) / generic_variable,
                                variable=generic_variable,
                                point=sp.Integer(0),
                            ),
                            right=sp.Integer(1),
                        ),
                    ),
                ),
            ),
        )

    numerator, denominator = sp.fraction(sp.together(expression))
    if point == sp.Integer(0) and denominator == variable and numerator.func == sp.sin:
        sine_argument = numerator.args[0]
        frequency = sp.simplify(sine_argument / variable)
        if not frequency.has(variable) and frequency != sp.Integer(0) and result == frequency:
            generic_variable = sp.Symbol("u", real=True)
            return (
                BackendDerivationStep(
                    rule="Normalize to the standard sine limit",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "Multiply and divide by the sine argument's coefficient, then use the "
                        "standard sine limit."
                    ),
                    verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    verification_detail="The normalized quotient has the standard limit one.",
                    notes=(
                        BackendMathNote(
                            label="Rewrite the quotient",
                            expression=BackendIdentity(
                                left=expression,
                                right=BackendProduct(
                                    factors=(
                                        frequency,
                                        BackendQuotient(
                                            numerator=sp.sin(sine_argument),
                                            denominator=sine_argument,
                                        ),
                                    )
                                ),
                            ),
                        ),
                        BackendMathNote(
                            label="Standard limit",
                            expression=BackendIdentity(
                                left=BackendLimit(
                                    expression=sp.sin(generic_variable) / generic_variable,
                                    variable=generic_variable,
                                    point=sp.Integer(0),
                                ),
                                right=sp.Integer(1),
                            ),
                        ),
                    ),
                ),
            )
    return ()


def derive_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive familiar limits with the shortest standard student method."""
    sine_derivation = _derive_sine_limit(expression, variable, point, direction, result)
    if sine_derivation:
        return sine_derivation
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )

    numerator, denominator = sp.fraction(sp.together(expression))
    if point not in {sp.oo, -sp.oo}:
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

    if point in {sp.oo, -sp.oo}:
        return _derive_infinite_limit(expression, variable, point, direction, result)
    return ()
