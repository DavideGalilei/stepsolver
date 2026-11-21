"""Factorial recurrence and growth comparison at infinity."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod


def derive_factorial_recurrence_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Cancel neighboring factorials before evaluating an infinite limit."""
    if point not in {sp.oo, -sp.oo} or not expression.has(sp.factorial(variable)):
        return ()
    simplified = sp.simplify(expression)
    if simplified == expression or simplified.has(sp.factorial):
        return ()
    try:
        simplified_result = sp.limit(simplified, variable, point)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if sp.simplify(simplified_result - result) != sp.Integer(0):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    simplified_limit = BackendLimit(
        expression=simplified,
        variable=variable,
        point=point,
        direction=direction,
    )
    generic = sp.Symbol("k", integer=True, positive=True)
    recurrence = BackendIdentity(
        left=sp.factorial(generic + 1),
        right=(generic + 1) * sp.factorial(generic),
    )
    return (
        BackendDerivationStep(
            rule="Apply the factorial recurrence",
            before=displayed,
            after=simplified_limit,
            explanation="Expand the larger neighboring factorial, then cancel the common one.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The factorial recurrence gives the simplified quotient exactly.",
            notes=(BackendMathNote(label="Factorial recurrence", expression=recurrence),),
        ),
        BackendDerivationStep(
            rule="Evaluate the simplified rational limit",
            before=simplified_limit,
            after=result,
            explanation="Compare the remaining leading powers as the variable tends to infinity.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The simplified rational expression has the displayed limit.",
        ),
    )


def derive_limit_at_infinity(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Compare leading terms in limits at infinity."""
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    if point not in {sp.oo, -sp.oo}:
        return ()
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
