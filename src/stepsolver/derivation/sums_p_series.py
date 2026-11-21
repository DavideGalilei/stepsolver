"""Standard, shifted and parity-restricted p-series."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

_GEOMETRIC_POWER_ARITY = 2


def _p_series_power(expression: sp.Basic, variable: sp.Symbol) -> int | None:
    return next(
        (
            power
            for power in range(1, 13)
            if sp.simplify(expression - variable ** (-power)) == sp.Integer(0)
        ),
        None,
    )


def derive_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use the standard p-series identity."""
    if lower != sp.Integer(1) or upper != sp.oo:
        return ()
    power = _p_series_power(expression, variable)
    if power is None:
        return ()
    generic_power = sp.Symbol("p", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=variable ** (-generic_power),
            variable=variable,
            lower=sp.Integer(1),
            upper=sp.oo,
        ),
        right=sp.zeta(generic_power),
    )
    convergence = sp.Gt(generic_power, sp.Integer(1), evaluate=False)
    sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if power <= 1:
        return (
            BackendDerivationStep(
                rule="Apply the p-series test",
                before=sigma,
                after=result,
                explanation="A p-series converges only when p > 1; here p = 1.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail="The p-series convergence criterion proves divergence.",
                notes=(BackendMathNote(label="Convergence condition", expression=convergence),),
            ),
        )
    zeta_value = sp.zeta(sp.Integer(power), evaluate=False)
    first_step = BackendDerivationStep(
        rule="Recognize a convergent p-series",
        before=sigma,
        after=zeta_value,
        explanation=f"This is a p-series with p = {power}, so it equals zeta({power}).",
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail="The p-series identity applies because the lower index is 1.",
        notes=(
            BackendMathNote(label="P-series identity", expression=identity),
            BackendMathNote(label="Convergence condition", expression=convergence),
        ),
    )
    if zeta_value == result:
        return (first_step,)
    return (
        first_step,
        BackendDerivationStep(
            rule=f"Use the exact value of zeta({power})",
            before=zeta_value,
            after=result,
            explanation="Substitute the known exact zeta value and simplify.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The special zeta value matches the exact symbolic result.",
            notes=(
                BackendMathNote(
                    label="Exact zeta value",
                    expression=BackendIdentity(left=zeta_value, right=result),
                ),
            ),
        ),
    )


def derive_shifted_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Shift the index into the standard p-series."""
    simplified = sp.simplify(expression)
    if upper != sp.oo or not simplified.is_Pow or len(simplified.args) != _GEOMETRIC_POWER_ARITY:
        return ()
    base, exponent = simplified.args
    power = next(
        (candidate for candidate in range(1, 13) if exponent == -sp.Integer(candidate)),
        None,
    )
    if power is None:
        return ()
    shift = sp.simplify(variable - base)
    if shift == sp.Integer(0) or variable in shift.free_symbols:
        return ()
    if sp.simplify(lower - shift) != sp.Integer(1):
        return ()
    shifted_variable = sp.Symbol("k", integer=True, positive=True)
    shifted_expression = shifted_variable ** (-power)
    shifted_sigma = BackendSigma(
        expression=shifted_expression,
        variable=shifted_variable,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    shift_step = BackendDerivationStep(
        rule="Shift the summation index",
        before=BackendSigma(
            expression=expression,
            variable=variable,
            lower=lower,
            upper=upper,
        ),
        after=shifted_sigma,
        explanation=(
            "Choose a new index that removes the constant shift. "
            "The lower bound becomes 1 and the upper bound remains infinity."
        ),
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail="The index substitution preserves every term and both bounds.",
        notes=(
            BackendMathNote(
                label="Index substitution",
                expression=BackendIdentity(left=shifted_variable, right=base),
            ),
        ),
    )
    return (
        shift_step,
        *derive_p_series(
            shifted_expression,
            shifted_variable,
            sp.Integer(1),
            sp.oo,
            result,
        ),
    )


def derive_even_odd_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    _result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the even or odd residue class of a convergent p-series."""
    if upper != sp.oo:
        return ()
    even_base = 2 * variable
    odd_base = 2 * variable + 1
    match = next(
        (
            (family, power, coefficient)
            for family, base, expected_lower in (
                ("even", even_base, sp.Integer(1)),
                ("odd", odd_base, sp.Integer(0)),
            )
            for power in range(2, 13)
            if lower == expected_lower
            and not (coefficient := sp.simplify(expression * base**power)).has(variable)
            and coefficient != sp.Integer(0)
        ),
        None,
    )
    if match is None:
        return ()
    family, power, coefficient = match
    residue_factor = (
        sp.Rational(1, 2**power) if family == "even" else sp.Integer(1) - sp.Rational(1, 2**power)
    )
    factor = sp.simplify(coefficient * residue_factor)
    explanation = (
        "Factor 2 from every even denominator, then use the p-series identity."
        if family == "even"
        else "Subtract the even-indexed p-series from the full positive p-series."
    )
    zeta_value = sp.zeta(sp.Integer(power))
    expected = sp.simplify(factor * zeta_value)
    index = sp.Symbol("k", integer=True, positive=True)
    full_series = BackendSigma(
        expression=index ** (-power),
        variable=index,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    return (
        BackendDerivationStep(
            rule=f"Extract the {family} terms of the p-series",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=expected,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The residue-class decomposition of the p-series is exact.",
            notes=(
                BackendMathNote(
                    label="Full p-series",
                    expression=BackendIdentity(left=full_series, right=zeta_value),
                ),
                BackendMathNote(
                    label=f"{family.title()}-term factor",
                    expression=BackendIdentity(
                        left=sp.Symbol("c", real=True),
                        right=factor,
                    ),
                ),
            ),
        ),
    )
