"""Dispatch limits in classroom method order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stepsolver.derivation.limits_algebraic import derive_finite_algebraic_limit
from stepsolver.derivation.limits_exponential import derive_variable_power_limit
from stepsolver.derivation.limits_infinity import (
    derive_factorial_recurrence_limit,
    derive_limit_at_infinity,
)
from stepsolver.derivation.limits_lhopital import derive_lhopital_limit
from stepsolver.derivation.limits_radicals import (
    derive_infinite_radical_conjugate,
    derive_radical_rationalization,
)

if TYPE_CHECKING:
    import sympy as sp

    from stepsolver.derivation.model import BackendDerivationStep


def derive_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive familiar limits with the shortest standard student method."""
    strategies = (
        derive_factorial_recurrence_limit,
        derive_lhopital_limit,
        derive_variable_power_limit,
        derive_radical_rationalization,
        derive_infinite_radical_conjugate,
        derive_finite_algebraic_limit,
        derive_limit_at_infinity,
    )
    for strategy in strategies:
        derivation = strategy(expression, variable, point, direction, result)
        if derivation:
            return derivation
    return ()
