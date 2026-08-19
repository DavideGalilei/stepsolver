"""Recognition and exact evaluation of symbolic series families."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True, kw_only=True)
class HarmonicSineSeries:
    """A matched series of the form ``a sin(n theta) / n``."""

    amplitude: sp.Basic
    angle: sp.Basic
    normalized_angle: sp.Basic
    value: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternatingPSeries:
    """A matched scalar multiple of ``(-1)^n / n^p``."""

    coefficient: sp.Basic
    power: int
    value: sp.Basic


def _principal_positive_angle(angle: sp.Basic) -> sp.Basic | None:
    """Normalize a provably real angle into the interval ``[0, 2 pi)``."""
    if angle.is_real is not True:
        return None
    pi_ratio = sp.simplify(angle / sp.pi)
    if pi_ratio.is_rational is True:
        return sp.simplify(sp.pi * sp.Mod(pi_ratio, sp.Integer(2)))
    positive = sp.simplify(sp.Gt(angle, sp.Integer(0)))
    below_period = sp.simplify(sp.Lt(angle, 2 * sp.pi))
    if positive is sp.true and below_period is sp.true:
        return angle
    return None


def match_harmonic_sine_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> HarmonicSineSeries | None:
    """Match and evaluate ``sum(a*sin(n*theta)/n, n, 1, oo)`` exactly."""
    if lower != sp.Integer(1) or upper != sp.oo:
        return None
    numerator = sp.simplify(expression * variable)
    amplitude, oscillation = numerator.as_independent(variable, as_Add=False)
    if amplitude.is_real is not True or oscillation.func is not sp.sin:
        return None
    argument = oscillation.args[0]
    if sp.simplify(argument.subs(variable, sp.Integer(0))) != sp.Integer(0):
        return None
    angle = sp.simplify(argument / variable)
    if variable in angle.free_symbols:
        return None
    normalized_angle = _principal_positive_angle(angle)
    if normalized_angle is None:
        return None
    value = (
        sp.Integer(0)
        if normalized_angle == sp.Integer(0)
        else sp.simplify(amplitude * (sp.pi - normalized_angle) / 2)
    )
    return HarmonicSineSeries(
        amplitude=amplitude,
        angle=angle,
        normalized_angle=normalized_angle,
        value=value,
    )


def match_alternating_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> AlternatingPSeries | None:
    """Match convergent scalar multiples of ``(-1)^n / n^p``."""
    if lower != sp.Integer(1) or upper != sp.oo:
        return None
    for power in range(1, 13):
        canonical = (-sp.Integer(1)) ** variable / variable**power
        coefficient = sp.simplify(expression / canonical)
        if variable in coefficient.free_symbols or coefficient.is_real is not True:
            continue
        if sp.simplify(expression - coefficient * canonical) != sp.Integer(0):
            continue
        eta_value = (
            sp.log(sp.Integer(2))
            if power == 1
            else sp.simplify(
                (1 - sp.Integer(2) ** (1 - power)) * sp.zeta(sp.Integer(power))
            )
        )
        return AlternatingPSeries(
            coefficient=coefficient,
            power=power,
            value=sp.simplify(-coefficient * eta_value),
        )
    return None
