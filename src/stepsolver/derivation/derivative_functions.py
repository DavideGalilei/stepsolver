"""Function and chain rules for elementary and special functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Callable

from stepsolver.derivation.derivative_steps import verified_derivative_steps


def derive_gamma_family(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate gamma and factorial expressions using the digamma function."""
    if expression.func not in {sp.gamma, sp.factorial} or len(expression.args) != 1:
        return ()
    argument = expression.args[0]
    if expression.func == sp.factorial:
        rewritten = sp.gamma(argument + 1)
        if sp.simplify(sp.diff(expression, variable) - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule="Rewrite the factorial with the gamma function",
                before=BackendDerivative(expression=expression, variable=variable),
                after=BackendDerivative(expression=rewritten, variable=variable),
                explanation="Use z! = Gamma(z + 1) so the derivative has a standard form.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The factorial-gamma identity holds on their common domain.",
                notes=(
                    BackendMathNote(
                        label="Factorial-gamma identity",
                        expression=BackendIdentity(left=expression, right=rewritten),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Differentiate gamma with the digamma function",
                before=BackendDerivative(expression=rewritten, variable=variable),
                after=result,
                explanation=(
                    "Use Gamma'(u) = Gamma(u) psi(u), then multiply by the inner derivative."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="The gamma derivative and chain rule give the exact result.",
            ),
        )
    return verified_derivative_steps(
        rule="Differentiate gamma with the digamma function",
        expression=expression,
        variable=variable,
        raw_derivative=result,
        result=result,
        explanation="Use Gamma'(u) = Gamma(u) psi(u), then apply the chain rule.",
        notes=(),
    )


def derive_absolute_value(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate an absolute value away from its nondifferentiable zeros."""
    if expression.func != sp.Abs or len(expression.args) != 1:
        return ()
    argument = expression.args[0]
    if sp.simplify(sp.diff(expression, variable) - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Differentiate the absolute value piecewise",
            before=BackendDerivative(expression=expression, variable=variable),
            after=result,
            explanation=(
                "Away from zeros of the inside expression, the slope is its derivative times "
                "the sign of that expression."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="The positive and negative branches have the displayed slopes.",
            introduced_constraints=(
                BackendStepConstraint(
                    explanation="The absolute value can fail to be differentiable at a zero.",
                    expression=BackendNotEqual(left=argument, right=sp.Integer(0)),
                ),
            ),
        ),
    )


def _negative_sine(argument: sp.Basic) -> sp.Basic:
    return -sp.sin(argument)


def _reciprocal(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(argument, -1, evaluate=False)


def _inverse_sine_derivative(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(1 - argument**2, sp.Rational(-1, 2), evaluate=False)


def _inverse_cosine_derivative(argument: sp.Basic) -> sp.Basic:
    return -_inverse_sine_derivative(argument)


def _inverse_tangent_derivative(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(1 + argument**2, -1, evaluate=False)


def _tangent_derivative(argument: sp.Basic) -> sp.Basic:
    return 1 + sp.tan(argument) ** 2


def _hyperbolic_tangent_derivative(argument: sp.Basic) -> sp.Basic:
    return 1 - sp.tanh(argument) ** 2


def _inverse_hyperbolic_sine_derivative(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(1 + argument**2, sp.Rational(-1, 2), evaluate=False)


_DERIVATIVE_PAIRS: tuple[
    tuple[Callable[[sp.Basic], sp.Basic], Callable[[sp.Basic], sp.Basic], str],
    ...,
] = (
    (sp.sin, sp.cos, "Differentiate the sine"),
    (sp.cos, _negative_sine, "Differentiate the cosine"),
    (sp.exp, sp.exp, "Differentiate the exponential"),
    (sp.log, _reciprocal, "Differentiate the logarithm"),
    (sp.tan, _tangent_derivative, "Differentiate the tangent"),
    (sp.asin, _inverse_sine_derivative, "Differentiate the inverse sine"),
    (sp.acos, _inverse_cosine_derivative, "Differentiate the inverse cosine"),
    (sp.atan, _inverse_tangent_derivative, "Differentiate the inverse tangent"),
    (sp.sinh, sp.cosh, "Differentiate the hyperbolic sine"),
    (sp.cosh, sp.sinh, "Differentiate the hyperbolic cosine"),
    (
        sp.tanh,
        _hyperbolic_tangent_derivative,
        "Differentiate the hyperbolic tangent",
    ),
    (
        sp.asinh,
        _inverse_hyperbolic_sine_derivative,
        "Differentiate the inverse hyperbolic sine",
    ),
)


def derive_function_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the derivative pair for the outer function and verify the chain rule."""
    for function, outer_derivative, direct_rule in _DERIVATIVE_PAIRS:
        if expression.func != function or len(expression.args) != 1:
            continue
        argument = expression.args[0]
        inner_derivative = sp.diff(argument, variable)
        outer_value = outer_derivative(argument)
        raw_derivative = (
            outer_value
            if inner_derivative == sp.Integer(1)
            else sp.Mul(outer_value, inner_derivative, evaluate=False)
        )
        is_direct = argument == variable
        notes: tuple[BackendMathNote, ...] = (
            BackendMathNote(
                label="Outer derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=function(sp.Symbol("u", real=True)),
                        variable=sp.Symbol("u", real=True),
                    ),
                    right=outer_derivative(sp.Symbol("u", real=True)),
                ),
            ),
        )
        if not is_direct:
            notes = (
                *notes,
                BackendMathNote(
                    label="Inner derivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(expression=argument, variable=variable),
                        right=inner_derivative,
                    ),
                ),
            )
        return verified_derivative_steps(
            rule=direct_rule if is_direct else "Apply the chain rule",
            expression=expression,
            variable=variable,
            raw_derivative=raw_derivative,
            result=result,
            explanation=(
                "Use the basic derivative pair."
                if is_direct
                else "Differentiate the outer function, then multiply by the inner derivative."
            ),
            notes=notes,
        )
    return ()
