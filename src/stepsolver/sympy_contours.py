"""Exact contour analysis for rational functions on parameterized circles."""

from dataclasses import dataclass

import sympy as sp

from stepsolver.errors import QueryError
from stepsolver.sympy_support import is_object_sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class ContourExpression:
    """Backend expressions that define one parameterized contour integral."""

    integrand: sp.Basic
    variable: sp.Symbol
    path: sp.Basic
    parameter: sp.Symbol
    lower: sp.Basic
    upper: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class RationalCircleContour:
    """An exact residue-theorem evaluation of one circular contour."""

    winding_number: int
    enclosed_residues: tuple[tuple[sp.Basic, sp.Basic], ...]
    residue_sum: sp.Basic
    value: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class _CircleGeometry:
    center: sp.Basic
    radius_squared: sp.Basic
    winding_number: int


def _circle_geometry(contour: ContourExpression) -> _CircleGeometry | None:
    center_pattern = sp.Wild("contour_center", exclude=[contour.parameter])
    radius_pattern = sp.Wild("contour_radius", exclude=[contour.parameter])
    frequency_pattern = sp.Wild("contour_frequency", exclude=[contour.parameter])
    match = contour.path.match(
        center_pattern + radius_pattern * sp.exp(sp.I * frequency_pattern * contour.parameter)
    )
    if match is None:
        return None
    center = match.get(center_pattern)
    radius = match.get(radius_pattern)
    frequency = match.get(frequency_pattern)
    if center is None or radius is None or frequency is None:
        return None
    if center.free_symbols or radius.free_symbols or frequency.free_symbols:
        return None
    winding = sp.simplify(frequency * (contour.upper - contour.lower) / (2 * sp.pi))
    radius_squared = sp.simplify(radius * sp.conjugate(radius))
    if (
        not isinstance(winding, sp.Integer)
        or int(winding) == 0
        or radius_squared.is_positive is not True
    ):
        return None
    return _CircleGeometry(
        center=center,
        radius_squared=radius_squared,
        winding_number=int(winding),
    )


def _pole_is_inside(pole: sp.Basic, geometry: _CircleGeometry) -> bool | None:
    offset = sp.simplify(pole - geometry.center)
    distance_squared = sp.simplify(offset * sp.conjugate(offset))
    boundary_test = sp.simplify(distance_squared - geometry.radius_squared)
    if boundary_test.is_zero is True:
        message = "the contour passes through a pole of the integrand"
        raise QueryError(message)
    if boundary_test.is_negative is True:
        return True
    if boundary_test.is_positive is True:
        return False
    return None


def _enclosed_residues(
    integrand: sp.Basic,
    variable: sp.Symbol,
    geometry: _CircleGeometry,
) -> tuple[tuple[sp.Basic, sp.Basic], ...] | None:
    roots = sp.solve(sp.denom(integrand), variable)
    if not is_object_sequence(roots):
        return None
    enclosed: list[tuple[sp.Basic, sp.Basic]] = []
    seen: set[sp.Basic] = set()
    for root in roots:
        if not isinstance(root, sp.Basic) or root in seen:
            continue
        seen.add(root)
        is_inside = _pole_is_inside(root, geometry)
        if is_inside is None:
            return None
        if is_inside:
            enclosed.append((root, sp.simplify(sp.residue(integrand, variable, root))))
    return tuple(enclosed)


def evaluate_rational_circle(
    contour: ContourExpression,
) -> RationalCircleContour | None:
    """Evaluate a rational integrand over an exact exponential circle when possible."""
    geometry = _circle_geometry(contour)
    rational_integrand = sp.cancel(contour.integrand)
    if geometry is None or rational_integrand.is_rational_function(contour.variable) is not True:
        return None
    enclosed = _enclosed_residues(
        rational_integrand,
        contour.variable,
        geometry,
    )
    if enclosed is None:
        return None
    residue_sum = sp.simplify(sum((residue for _, residue in enclosed), sp.Integer(0)))
    value = sp.simplify(2 * sp.pi * sp.I * geometry.winding_number * residue_sum)
    return RationalCircleContour(
        winding_number=geometry.winding_number,
        enclosed_residues=enclosed,
        residue_sum=residue_sum,
        value=value,
    )
