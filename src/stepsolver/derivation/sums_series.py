"""Classroom order for closed-form summation strategies."""

from __future__ import annotations

from stepsolver.derivation.sums_geometric import (
    derive_differentiated_geometric_series,
    derive_geometric_series,
    derive_second_moment_geometric_series,
)
from stepsolver.derivation.sums_p_series import (
    derive_even_odd_p_series,
    derive_p_series,
    derive_shifted_p_series,
)
from stepsolver.derivation.sums_telescoping import derive_telescoping_rational_sum
from stepsolver.derivation.sums_trigonometric import derive_gregory_leibniz_series

CLOSED_FORM_SUM_STRATEGIES = (
    derive_p_series,
    derive_shifted_p_series,
    derive_even_odd_p_series,
    derive_geometric_series,
    derive_differentiated_geometric_series,
    derive_second_moment_geometric_series,
    derive_telescoping_rational_sum,
    derive_gregory_leibniz_series,
)
