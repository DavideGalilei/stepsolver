"""Focused backend derivation strategy tests."""

import pytest
import sympy as sp

from stepsolver.sympy_derivation import derive_reciprocal_quadratic_integral


def test_integral_strategy_declines_unsupported_rational_forms() -> None:
    """Only constant-over-positive-quadratic integrands should use this strategy."""
    variable = sp.Symbol("x", real=True)
    assert (
        derive_reciprocal_quadratic_integral(
            variable / (variable**2 + 1),
            variable,
            sp.log(variable**2 + 1) / 2,
        )
        == ()
    )
    assert (
        derive_reciprocal_quadratic_integral(
            1 / (variable + 1),
            variable,
            sp.log(variable + 1),
        )
        == ()
    )
    assert (
        derive_reciprocal_quadratic_integral(
            1 / (variable**2 - 1),
            variable,
            sp.log((variable - 1) / (variable + 1)) / 2,
        )
        == ()
    )


def test_integral_strategy_rejects_an_incorrect_expected_result() -> None:
    """A derivation must agree with the exact backend antiderivative."""
    variable = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="differs from the exact result"):
        derive_reciprocal_quadratic_integral(
            1 / (variable**2 + 1),
            variable,
            variable,
        )
