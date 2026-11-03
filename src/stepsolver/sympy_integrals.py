"""Recognition and exact evaluation of definite-integral families."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import ceil, floor

import sympy as sp

_MAX_SINE_PIECES = 8
_POWER_ARITY = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class SineSquareRootIntegral:
    """A matched integral whose integrand is the principal root of ``sin(x)^2``."""

    points: tuple[sp.Basic, ...]
    signed_integrands: tuple[sp.Basic, ...]
    piece_values: tuple[sp.Basic, ...]
    value: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class ReciprocalLinearIntegral:
    """A constant over a linear denominator, integrated over the real domain."""

    coefficient: sp.Basic
    denominator: sp.Basic
    denominator_derivative: sp.Basic
    root: sp.Basic
    antiderivative: sp.Basic
    result: sp.Basic


def match_reciprocal_linear_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
) -> ReciprocalLinearIntegral | None:
    """Match ``c / (a*x + b)`` and return its real-domain logarithmic antiderivative."""
    numerator, denominator = sp.fraction(sp.together(integrand))
    if (
        numerator.free_symbols
        or numerator.is_real is not True
        or denominator.free_symbols != {variable}
    ):
        return None
    try:
        polynomial = sp.Poly(denominator, variable)
    except sp.PolynomialError:
        return None
    if polynomial.degree() != 1:
        return None
    leading = polynomial.coeff_monomial(variable)
    constant = polynomial.coeff_monomial(1)
    if leading == sp.Integer(0) or leading.is_real is not True or constant.is_real is not True:
        return None
    coefficient = sp.simplify(numerator / leading)
    denominator_derivative = sp.diff(denominator, variable)
    if sp.simplify(denominator_derivative - leading) != sp.Integer(0):
        return None
    root = sp.simplify(-constant / leading)
    logarithm = sp.log(sp.Abs(denominator))
    antiderivative = (
        logarithm
        if coefficient == sp.Integer(1)
        else sp.Mul(coefficient, logarithm, evaluate=False)
    )
    integration_constant = sp.Symbol("C")
    return ReciprocalLinearIntegral(
        coefficient=coefficient,
        denominator=denominator,
        denominator_derivative=denominator_derivative,
        root=root,
        antiderivative=antiderivative,
        result=sp.Add(antiderivative, integration_constant, evaluate=False),
    )


def _pi_rational_points(
    lower: sp.Basic,
    upper: sp.Basic,
) -> tuple[tuple[sp.Basic, ...], tuple[Fraction, ...]] | None:
    """Return endpoints and intervening integer multiples of pi."""
    lower_ratio = sp.simplify(lower / sp.pi)
    upper_ratio = sp.simplify(upper / sp.pi)
    if not isinstance(lower_ratio, sp.Rational) or not isinstance(upper_ratio, sp.Rational):
        return None
    lower_fraction = Fraction(str(lower_ratio))
    upper_fraction = Fraction(str(upper_ratio))
    if lower_fraction >= upper_fraction:
        return None
    first_zero = floor(lower_fraction) + 1
    zero_limit = ceil(upper_fraction)
    zero_indices = range(first_zero, zero_limit)
    points = (lower, *(sp.Integer(index) * sp.pi for index in zero_indices), upper)
    if len(points) - 1 > _MAX_SINE_PIECES:
        return None
    ratio_points = (
        lower_fraction,
        *(Fraction(index) for index in zero_indices),
        upper_fraction,
    )
    return points, ratio_points


def match_sine_square_root_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> SineSquareRootIntegral | None:
    """Match and evaluate ``sqrt(1-cos(x)^2)`` on manageable pi-rational intervals."""
    sine = sp.sin(variable)
    absolute_sine = sp.Abs(sine)
    is_trigonometric_root = (
        integrand.func == sp.Pow
        and len(integrand.args) == _POWER_ARITY
        and integrand.args[1] == sp.Rational(1, 2)
        and (integrand.has(sine) or integrand.has(sp.cos(variable)))
    )
    if integrand != absolute_sine and not is_trigonometric_root:
        return None
    if sp.simplify(integrand - absolute_sine) != sp.Integer(0):
        return None
    interval = _pi_rational_points(lower, upper)
    if interval is None:
        return None
    points, ratio_points = interval
    signed_integrands: list[sp.Basic] = []
    piece_values: list[sp.Basic] = []
    for (piece_lower, piece_upper), (ratio_lower, ratio_upper) in zip(
        pairwise(points),
        pairwise(ratio_points),
        strict=True,
    ):
        interval_index = floor((ratio_lower + ratio_upper) / 2)
        sign = sp.Integer(1) if interval_index % 2 == 0 else sp.Integer(-1)
        signed_integrand = sign * sp.sin(variable)
        piece_value = sp.integrate(signed_integrand, (variable, piece_lower, piece_upper))
        if piece_value.has(sp.Integral):
            return None
        signed_integrands.append(signed_integrand)
        piece_values.append(sp.simplify(piece_value))
    values = tuple(piece_values)
    return SineSquareRootIntegral(
        points=points,
        signed_integrands=tuple(signed_integrands),
        piece_values=values,
        value=sp.simplify(sum(values, start=sp.Integer(0))),
    )
