"""Linear equation data and verified SymPy solution mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.derivation.model import BackendSystem
from stepsolver.sympy_support import is_object_mapping, is_object_sequence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def system_display(equations: Sequence[sp.Basic]) -> BackendSystem:
    """Render equations as a system."""
    return BackendSystem(equations=tuple(equations))


def relations_display(relations: tuple[sp.Equality, ...]) -> sp.Equality | BackendSystem:
    """Use a standalone equality when only one variable is solved."""
    if len(relations) == 1:
        return relations[0]
    return system_display(relations)


def solution_mapping(
    backend_value: object,
    variables: Sequence[sp.Symbol],
    *,
    require_every_variable: bool = True,
) -> dict[sp.Symbol, sp.Basic] | None:
    """Validate the single solution mapping returned by SymPy."""
    if not is_object_sequence(backend_value) or len(backend_value) != 1:
        return None
    candidate = backend_value[0]
    if not is_object_mapping(candidate):
        return None
    solution: dict[sp.Symbol, sp.Basic] = {}
    for variable, value in candidate.items():
        if not isinstance(variable, sp.Symbol) or not isinstance(value, sp.Basic):
            return None
        solution[variable] = value
    if require_every_variable and any(variable not in solution for variable in variables):
        return None
    return solution


def solution_relations(
    solution: Mapping[sp.Symbol, sp.Basic],
    variables: Sequence[sp.Symbol],
) -> tuple[sp.Equality, ...]:
    """Render the solved variables in their original order."""
    return tuple(
        sp.Eq(variable, value, evaluate=False)
        for variable in variables
        if (value := solution.get(variable)) is not None
    )


def linear_row(
    equation: sp.Equality,
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Basic, ...] | None:
    """Return coefficients in variable order followed by the right-hand side."""
    difference = sp.expand(equation.lhs - equation.rhs)
    try:
        polynomial = sp.Poly(difference, variables)
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1:
        return None
    return (
        *(polynomial.coeff_monomial(variable) for variable in variables),
        -polynomial.coeff_monomial(1),
    )
