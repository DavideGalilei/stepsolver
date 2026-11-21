"""Geometric sums and their differentiated moments."""

from __future__ import annotations

from typing import Literal

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendExpression,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

_GEOMETRIC_POWER_ARITY = 2


def derive_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use the geometric identity with the applicable convergence condition."""
    if (
        lower.is_integer is not True
        or not expression.is_Pow
        or len(expression.args) != _GEOMETRIC_POWER_ARITY
    ):
        return ()
    ratio, exponent = expression.args
    if exponent != variable:
        return ()
    generic_ratio = sp.Symbol("r", real=True)
    generic_lower = sp.Symbol("m", integer=True, nonnegative=True)
    identity_lower = sp.Integer(0) if lower == sp.Integer(0) else generic_lower
    condition: BackendExpression
    if upper == sp.oo:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=generic_ratio**variable,
                variable=variable,
                lower=identity_lower,
                upper=sp.oo,
            ),
            right=generic_ratio**identity_lower / (1 - generic_ratio),
        )
        condition = sp.Lt(sp.Abs(generic_ratio), sp.Integer(1))
        ratio_magnitude = sp.Abs(ratio)
        converges = sp.simplify(sp.Lt(ratio_magnitude, sp.Integer(1)))
        if converges is sp.true:
            explanation = "The common ratio satisfies |r| < 1, so the series converges."
            rule = "Apply the infinite geometric-series identity"
            after: BackendExpression = result
            verification_detail = "The geometric-series formula gives the exact result."
        else:
            explanation = (
                "An infinite geometric series converges only when |r| < 1. "
                "This common ratio does not satisfy that condition."
            )
            rule = "Apply the geometric-series convergence test"
            after = sp.Ge(ratio_magnitude, sp.Integer(1), evaluate=False)
            verification_detail = "The convergence criterion was applied to the common ratio."
    elif upper.is_integer is True:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=generic_ratio**variable,
                variable=variable,
                lower=identity_lower,
                upper=upper,
            ),
            right=(generic_ratio**identity_lower - generic_ratio ** (upper + 1))
            / (1 - generic_ratio),
        )
        condition = BackendNotEqual(left=generic_ratio, right=sp.Integer(1))
        explanation = "Use the finite geometric-series identity and substitute the ratio."
        rule = "Apply the finite geometric-series identity"
        after = result
        verification_detail = "The geometric-series formula gives the exact result."
    else:
        return ()
    return (
        BackendDerivationStep(
            rule=rule,
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=after,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=verification_detail,
            notes=(
                BackendMathNote(label="Identity", expression=identity),
                BackendMathNote(label="Required condition", expression=condition),
                BackendMathNote(
                    label="Common ratio",
                    expression=BackendIdentity(left=generic_ratio, right=ratio),
                ),
                *(
                    (
                        BackendMathNote(
                            label="Starting index",
                            expression=BackendIdentity(left=generic_lower, right=lower),
                        ),
                    )
                    if lower != sp.Integer(0)
                    else ()
                ),
            ),
        ),
    )


def derive_differentiated_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Sum c*n*r^n by differentiating the geometric-series identity."""
    return _derive_weighted_geometric_series(expression, variable, lower, upper, result, degree=1)


def derive_second_moment_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Sum c*n**2*r**n by differentiating the geometric identity twice."""
    return _derive_weighted_geometric_series(expression, variable, lower, upper, result, degree=2)


def _weighted_geometric_term(
    expression: sp.Basic,
    variable: sp.Symbol,
    degree: Literal[1, 2],
) -> tuple[sp.Basic, sp.Basic] | None:
    power_match = next(
        (
            match
            for factor in expression.as_ordered_factors()
            if (match := _variable_power(factor, variable)) is not None
        ),
        None,
    )
    if power_match is None:
        return None
    base, exponent = power_match
    slope = sp.diff(exponent, variable)
    if slope.has(variable):
        return None
    ratio = sp.simplify(base**slope)
    if sp.simplify(sp.Lt(sp.Abs(ratio), sp.Integer(1))) is not sp.true:
        return None
    coefficient = sp.simplify(expression / (variable**degree * ratio**variable))
    if coefficient.has(variable):
        return None
    return ratio, coefficient


def _geometric_moment(ratio: sp.Basic, degree: Literal[1, 2]) -> sp.Basic:
    if degree == 1:
        return ratio / (1 - ratio) ** 2
    return ratio * (1 + ratio) / (1 - ratio) ** 3


def _derive_weighted_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
    *,
    degree: Literal[1, 2],
) -> tuple[BackendDerivationStep, ...]:
    if lower != sp.Integer(1) or upper != sp.oo:
        return ()
    match = _weighted_geometric_term(expression, variable, degree)
    if match is None:
        return ()
    ratio, coefficient = match
    if sp.simplify(coefficient * _geometric_moment(ratio, degree)) != result:
        return ()

    generic_ratio = sp.Symbol("r", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=variable**degree * generic_ratio**variable,
            variable=variable,
            lower=lower,
            upper=upper,
        ),
        right=_geometric_moment(generic_ratio, degree),
    )
    if degree == 1:
        rule = "Differentiate the geometric-series identity"
        explanation = (
            "Differentiate Σrⁿ = 1/(1-r), multiply by r, then substitute the common "
            "ratio and outside coefficient."
        )
        detail = "The differentiated geometric identity gives the exact sum."
        label = "Weighted geometric identity"
    else:
        rule = "Differentiate the geometric-series identity twice"
        explanation = (
            "Differentiate the geometric series twice, then combine the resulting "
            "first- and second-derivative terms and substitute the common ratio."
        )
        detail = "The second-moment geometric identity gives the exact sum."
        label = "Second-moment identity"
    return (
        BackendDerivationStep(
            rule=rule,
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=detail,
            notes=(
                BackendMathNote(label=label, expression=identity),
                BackendMathNote(
                    label="Common ratio",
                    expression=BackendIdentity(left=generic_ratio, right=ratio),
                ),
                BackendMathNote(
                    label="Convergence condition",
                    expression=sp.Lt(sp.Abs(generic_ratio), sp.Integer(1)),
                ),
            ),
        ),
    )


def _variable_power(
    factor: sp.Basic,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic] | None:
    if not factor.is_Pow or len(factor.args) != _GEOMETRIC_POWER_ARITY:
        return None
    base, exponent = factor.args
    if exponent.has(variable):
        return base, exponent
    if base.is_Pow and len(base.args) == _GEOMETRIC_POWER_ARITY:
        nested_base, nested_exponent = base.args
        combined_exponent = sp.simplify(nested_exponent * exponent)
        if combined_exponent.has(variable):
            return nested_base, combined_exponent
    return None
