"""Dispatch differentiation methods in classroom order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stepsolver.derivation.derivative_algebra import (
    derive_monomial,
    derive_product,
    derive_quotient,
    derive_sum,
)
from stepsolver.derivation.derivative_functions import (
    derive_absolute_value,
    derive_function_chain,
    derive_gamma_family,
)
from stepsolver.derivation.derivative_powers import derive_general_power, derive_power_chain

if TYPE_CHECKING:
    import sympy as sp

    from stepsolver.derivation.model import BackendDerivationStep


def derive_derivative(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a first derivative using the most specific familiar rule."""
    strategies = (
        derive_gamma_family,
        derive_absolute_value,
        derive_general_power,
        derive_quotient,
        derive_monomial,
        derive_sum,
        derive_product,
        derive_power_chain,
        derive_function_chain,
    )
    for strategy in strategies:
        derivation = strategy(expression, variable, result)
        if derivation:
            return derivation
    return ()
