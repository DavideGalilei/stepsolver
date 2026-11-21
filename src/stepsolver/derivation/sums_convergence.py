"""Classroom order for convergence and divergence strategies."""

from __future__ import annotations

from stepsolver.derivation.sums_alternating import derive_alternating_p_series
from stepsolver.derivation.sums_divergence import (
    derive_nth_root_power_divergence,
    derive_nth_term_divergence,
)
from stepsolver.derivation.sums_trigonometric import derive_harmonic_sine_series

CONVERGENCE_SUM_STRATEGIES = (
    derive_harmonic_sine_series,
    derive_alternating_p_series,
    derive_nth_root_power_divergence,
    derive_nth_term_divergence,
)
