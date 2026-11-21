"""Choose an integral derivation while preserving classroom method precedence."""

import sympy as sp

from stepsolver.derivation.definite_integrals import (
    derive_definite_integral,
    derive_dirichlet_integral,
    derive_improper_integral,
)
from stepsolver.derivation.integrals_benchmark import derive_advanced_substitution_integral
from stepsolver.derivation.integrals_elementary import (
    derive_basic_antiderivative,
    derive_constant_multiple_integral,
    derive_polynomial_sum_integral,
)
from stepsolver.derivation.integrals_parts import derive_integration_by_parts
from stepsolver.derivation.integrals_radical import (
    derive_shifted_semicircle_integral,
    derive_square_root_rational_integral,
)
from stepsolver.derivation.integrals_rational import derive_partial_fraction_integral
from stepsolver.derivation.integrals_structural import (
    derive_cyclic_exponential_trig_integral,
    derive_inverse_function_by_parts,
    derive_inverse_tangent_substitution,
    derive_trig_power_substitution,
)
from stepsolver.derivation.integrals_substitution import (
    derive_function_substitution_integral,
    derive_log_derivative_integral,
)
from stepsolver.derivation.integrals_transcendental import (
    derive_gaussian_antiderivative,
    derive_inverse_hyperbolic_integral,
    derive_trigonometric_power_integral,
)
from stepsolver.derivation.model import BackendDerivationStep
from stepsolver.derivation.reciprocal_quadratic import derive_reciprocal_quadratic_integral

# Earlier direct rules and recognizable substitutions take precedence over
# methods that rewrite the integrand or introduce auxiliary integrals.
_INDEFINITE_METHODS = (
    derive_basic_antiderivative,
    derive_log_derivative_integral,
    derive_polynomial_sum_integral,
    derive_constant_multiple_integral,
    derive_function_substitution_integral,
    derive_trig_power_substitution,
    derive_inverse_tangent_substitution,
    derive_square_root_rational_integral,
    derive_shifted_semicircle_integral,
    derive_inverse_hyperbolic_integral,
    derive_trigonometric_power_integral,
    derive_inverse_function_by_parts,
    derive_cyclic_exponential_trig_integral,
    derive_integration_by_parts,
    derive_advanced_substitution_integral,
    derive_gaussian_antiderivative,
    derive_reciprocal_quadratic_integral,
    derive_partial_fraction_integral,
)


def build_antiderivative_steps(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Return the first applicable derivation in classroom method order."""
    for strategy in _INDEFINITE_METHODS:
        derivation = strategy(integrand, variable, result)
        if derivation:
            return derivation
    return ()


def build_definite_integral_steps(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Prefer domain-specific and improper methods before endpoint evaluation."""
    derivation = derive_dirichlet_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if derivation:
        return derivation
    derivation = derive_improper_integral(integrand, variable, lower, upper, result)
    if derivation:
        return derivation
    return derive_definite_integral(integrand, variable, lower, upper, result)
