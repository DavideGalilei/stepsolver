"""Limits of variable powers and exponential definitions."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_POWER_ARITY = 2


def derive_variable_power_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Handle variable powers using the exponential of a logarithm."""
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    if expression == variable**variable and point == sp.Integer(0) and direction == "+":
        rewritten = sp.exp(variable * sp.log(variable))
        exponent_limit = BackendLimit(
            expression=variable * sp.log(variable),
            variable=variable,
            point=point,
            direction=direction,
        )
        if result != sp.Integer(1):
            return ()
        return (
            BackendDerivationStep(
                rule="Rewrite the variable power exponentially",
                before=displayed,
                after=BackendLimit(
                    expression=rewritten,
                    variable=variable,
                    point=point,
                    direction=direction,
                ),
                explanation="For positive x, write xˣ as e^(x log(x)).",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The exponential-logarithm identity holds for x > 0.",
                notes=(BackendMathNote(label="Exponent limit", expression=exponent_limit),),
            ),
            BackendDerivationStep(
                rule="Evaluate the exponent limit",
                before=exponent_limit,
                after=result,
                explanation="Since x log(x) tends to zero, continuity of e^t gives e^0 = 1.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail=(
                    "The standard x log(x) limit and continuity of the exponential apply."
                ),
            ),
        )
    if point == sp.oo and expression.is_Pow and len(expression.args) == _POWER_ARITY:
        base, exponent = expression.args
        increment = sp.simplify((base - 1) * variable)
        exponent_scale = sp.simplify(exponent / variable)
        expected = sp.exp(sp.simplify(increment * exponent_scale))
        if (
            not increment.has(variable)
            and not exponent_scale.has(variable)
            and exponent_scale != sp.Integer(0)
            and sp.simplify(result - expected) == sp.Integer(0)
        ):
            generic = sp.Symbol("a", real=True)
            return (
                BackendDerivationStep(
                    rule="Use the exponential-definition limit",
                    before=displayed,
                    after=result,
                    explanation=(
                        "This has the form (1+a/x)^(b x), whose limit is e^(a b). "
                        "Substitute the constant increment and exponent scale."
                    ),
                    verification_method=VerificationMethod.BACKEND_IDENTITY,
                    verification_detail="The defining exponential limit gives the exact value.",
                    notes=(
                        BackendMathNote(
                            label="General identity",
                            expression=BackendIdentity(
                                left=BackendLimit(
                                    expression=(1 + generic / variable) ** variable,
                                    variable=variable,
                                    point=sp.oo,
                                ),
                                right=sp.exp(generic),
                            ),
                        ),
                    ),
                ),
            )
    return ()
