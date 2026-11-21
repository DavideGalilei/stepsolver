"""Differentiation of composite and variable powers."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.derivative_steps import verified_derivative_steps
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendIdentity,
    BackendMathNote,
    BackendProduct,
    BackendQuotient,
    BackendStepConstraint,
    BackendSum,
)

_BINARY_ARITY = 2


def derive_general_power(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate f(x)^g(x) by taking logarithms first."""
    if not expression.is_Pow or len(expression.args) != _BINARY_ARITY:
        return ()
    base, exponent = expression.args
    if not base.has(variable) or not exponent.has(variable):
        return ()
    base_derivative = sp.diff(base, variable)
    exponent_derivative = sp.diff(exponent, variable)
    logarithmic_derivative = exponent_derivative * sp.log(base) + exponent * base_derivative / base
    raw_derivative = expression * logarithmic_derivative
    generic_base = sp.Function("f")(variable)
    generic_exponent = sp.Function("g")(variable)
    identity = BackendIdentity(
        left=BackendDerivative(
            expression=generic_base**generic_exponent,
            variable=variable,
        ),
        right=BackendProduct(
            factors=(
                generic_base**generic_exponent,
                BackendSum(
                    terms=(
                        BackendProduct(
                            factors=(
                                BackendDerivative(
                                    expression=generic_exponent,
                                    variable=variable,
                                ),
                                sp.log(generic_base),
                            ),
                        ),
                        BackendQuotient(
                            numerator=BackendProduct(
                                factors=(
                                    generic_exponent,
                                    BackendDerivative(
                                        expression=generic_base,
                                        variable=variable,
                                    ),
                                ),
                            ),
                            denominator=generic_base,
                        ),
                    ),
                ),
            ),
        ),
    )
    return verified_derivative_steps(
        rule="Use logarithmic differentiation",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Take logarithms so the variable exponent becomes a product, differentiate, "
            "then multiply by the original power."
        ),
        notes=(BackendMathNote(label="General power rule", expression=identity),),
        show_simplification=True,
        introduced_constraints=(
            BackendStepConstraint(
                explanation="Logarithmic differentiation uses the real logarithm of the base.",
                expression=sp.Gt(base, sp.Integer(0), evaluate=False),
            ),
        ),
    )


def derive_power_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the power and chain rules to a composite base."""
    if not expression.is_Pow or len(expression.args) != _BINARY_ARITY:
        return ()
    base, exponent = expression.args
    if exponent.has(variable):
        return ()
    base_derivative = sp.diff(base, variable)
    outer_power_derivative = sp.Mul(
        exponent,
        sp.Pow(base, exponent - 1, evaluate=False),
        evaluate=False,
    )
    raw_derivative = (
        outer_power_derivative
        if base_derivative == sp.Integer(1)
        else sp.Mul(outer_power_derivative, base_derivative, evaluate=False)
    )
    return verified_derivative_steps(
        rule="Apply the power and chain rules",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=("Differentiate the outer power, then multiply by the derivative of its base."),
        notes=(
            BackendMathNote(
                label="Base derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=base, variable=variable),
                    right=base_derivative,
                ),
            ),
        ),
    )
