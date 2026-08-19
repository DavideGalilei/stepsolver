"""Focused recognition tests for reusable infinite-series families."""

import sympy as sp

from stepsolver.sympy_series import match_alternating_p_series, match_harmonic_sine_series


def test_harmonic_sine_matcher_rejects_noncanonical_series() -> None:
    """Bounds, phase shifts, nonlinear phases, and nonreal angles must not be guessed."""
    variable = sp.Symbol("n", integer=True, positive=True)
    assert (
        match_harmonic_sine_series(
            sp.sin(variable) / variable,
            variable,
            sp.Integer(0),
            sp.oo,
        )
        is None
    )
    assert (
        match_harmonic_sine_series(
            sp.cos(variable) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
    assert (
        match_harmonic_sine_series(
            sp.sin(variable + 1) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
    assert (
        match_harmonic_sine_series(
            sp.sin(variable**2) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
    assert (
        match_harmonic_sine_series(
            sp.sin(sp.I * variable) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
    unspecified_angle = sp.Symbol("z")
    assert (
        match_harmonic_sine_series(
            sp.sin(unspecified_angle * variable) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
    assert (
        match_harmonic_sine_series(
            sp.sin((2 * sp.pi + sp.sqrt(sp.Integer(2))) * variable) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )


def test_alternating_p_matcher_rejects_wrong_bounds_and_nonmatching_terms() -> None:
    """Leibniz identities should only claim the exact p-series family from index one."""
    variable = sp.Symbol("n", integer=True, positive=True)
    assert (
        match_alternating_p_series(
            (-sp.Integer(1)) ** variable / variable,
            variable,
            sp.Integer(0),
            sp.oo,
        )
        is None
    )
    assert (
        match_alternating_p_series(
            (-sp.Integer(1)) ** variable / (variable + 1),
            variable,
            sp.Integer(1),
            sp.oo,
        )
        is None
    )
