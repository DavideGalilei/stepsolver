"""Factorial-series summation strategies."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendIdentity,
    BackendMathNote,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

_SQUARE_POWER = 2
_GEOMETRIC_POWER_ARITY = 2


def _factorial_series_base(expression: sp.Basic, variable: sp.Symbol) -> sp.Basic | None:
    """Match ``base**n / n!`` with a constant base."""
    numerator = sp.simplify(expression * sp.factorial(variable))
    if numerator == sp.Integer(1):
        return sp.Integer(1)
    if not numerator.is_Pow or len(numerator.args) != _GEOMETRIC_POWER_ARITY:
        return None
    base, exponent = numerator.args
    if exponent != variable or base.has(variable):
        return None
    return base


def _derive_factorial_series_tail(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a tail of the exponential power series with explicit omitted terms."""
    if upper != sp.oo or lower.is_integer is not True or lower.is_nonnegative is not True:
        return ()
    base = _factorial_series_base(expression, variable)
    if base is None:
        return ()
    full_value = sp.exp(base)
    generic_base = sp.Symbol("x", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=generic_base**variable / sp.factorial(variable),
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        ),
        right=sp.exp(generic_base),
    )
    tail = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if lower == sp.Integer(0):
        if sp.simplify(full_value - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule="Apply the exponential-series identity",
                before=tail,
                after=full_value,
                explanation="This is the Maclaurin series for the exponential function.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail="The exponential power-series identity applies exactly.",
                notes=(BackendMathNote(label="Identity", expression=identity),),
            ),
        )

    prefix_upper = lower - 1
    prefix_sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=sp.Integer(0),
        upper=prefix_upper,
    )
    prefix_value = sp.summation(expression, (variable, sp.Integer(0), prefix_upper))
    expected = sp.simplify(full_value - prefix_value)
    if sp.simplify(expected - result) != sp.Integer(0):
        return ()
    rewritten_tail = BackendDifference(left=full_value, right=prefix_sigma)
    evaluated_tail = BackendDifference(left=full_value, right=prefix_value)
    return (
        BackendDerivationStep(
            rule="Subtract the omitted exponential-series terms",
            before=tail,
            after=rewritten_tail,
            explanation=(
                "Start with the full exponential series at n = 0, then subtract every term "
                f"before n = {lower}."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="Removing the finite prefix leaves exactly the requested tail.",
            notes=(BackendMathNote(label="Exponential-series identity", expression=identity),),
        ),
        BackendDerivationStep(
            rule="Evaluate the omitted finite terms",
            before=rewritten_tail,
            after=evaluated_tail,
            explanation=(
                "Evaluate the factorials in the finite prefix and combine those rational terms."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The finite prefix was evaluated exactly.",
            notes=(
                BackendMathNote(
                    label="Finite prefix",
                    expression=BackendIdentity(left=prefix_sigma, right=prefix_value),
                ),
            ),
        ),
    )


def _factorial_parity_family(slope: sp.Basic, offset: sp.Basic) -> str | None:
    """Identify whether a linear factorial index selects even or odd terms."""
    match (slope, offset):
        case (value, parity) if value == sp.Integer(_SQUARE_POWER) and parity == sp.Integer(0):
            return "even"
        case (value, parity) if value == sp.Integer(_SQUARE_POWER) and parity == sp.Integer(1):
            return "odd"
        case _:
            return None


def _factorial_power_base(
    numerator: sp.Basic,
    factorial_index: sp.Basic,
    variable: sp.Symbol,
) -> sp.Basic | None:
    """Extract the constant base from ``base**factorial_index`` or implicit one."""
    if numerator == sp.Integer(1):
        return sp.Integer(1)
    if not numerator.is_Pow or len(numerator.args) != _GEOMETRIC_POWER_ARITY:
        return None
    base, exponent = numerator.args
    if exponent != factorial_index or base.has(variable):
        return None
    return base


def _derive_factorial_parity_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Recognize the even and odd factorial subseries of the exponential."""
    if lower != sp.Integer(0) or upper != sp.oo:
        return ()
    numerator, denominator = sp.fraction(sp.together(expression))
    if denominator.func != sp.factorial or len(denominator.args) != 1:
        return ()
    factorial_index = sp.expand(denominator.args[0])
    slope = sp.diff(factorial_index, variable)
    offset = sp.simplify(factorial_index.subs(variable, sp.Integer(0)))
    family = _factorial_parity_family(slope, offset)
    if family is None:
        return ()
    function = sp.cosh if family == "even" else sp.sinh
    rule = f"Apply the {family} exponential-series identity"
    base = _factorial_power_base(numerator, factorial_index, variable)
    if base is None:
        return ()
    expected = function(base)
    if sp.simplify(expected - result) != sp.Integer(0):
        return ()
    generic = sp.Symbol("x", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=generic**factorial_index / sp.factorial(factorial_index),
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        ),
        right=function(generic),
    )
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
            explanation=(
                f"Keep the {family}-indexed terms of the exponential power series; "
                f"they form the {function.__name__} series."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The corresponding factorial subseries identity applies exactly.",
            notes=(BackendMathNote(label="Identity", expression=identity),),
        ),
    )


FACTORIAL_SUM_STRATEGIES = (
    _derive_factorial_series_tail,
    _derive_factorial_parity_series,
)
