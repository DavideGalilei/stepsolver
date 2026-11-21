"""Choose the elimination method for a linear system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stepsolver.derivation.systems_elimination import derive_larger_linear_system
from stepsolver.derivation.systems_two_by_two import derive_two_by_two_linear_system

_MIN_SYSTEM_SIZE = 2


_TWO_BY_TWO_SIZE = 2


if TYPE_CHECKING:
    import sympy as sp

    from stepsolver.derivation.model import BackendDerivationStep


def derive_linear_system(
    equations: tuple[sp.Equality, ...],
    variables: tuple[sp.Symbol, ...],
    backend_value: object,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a linear system with elimination and back-substitution."""
    if len(equations) == _TWO_BY_TWO_SIZE and len(variables) == _TWO_BY_TWO_SIZE:
        return derive_two_by_two_linear_system(
            (equations[0], equations[1]),
            (variables[0], variables[1]),
            backend_value,
        )
    if len(equations) < _MIN_SYSTEM_SIZE or len(variables) < _MIN_SYSTEM_SIZE:
        return ()
    return derive_larger_linear_system(equations, variables, backend_value)
