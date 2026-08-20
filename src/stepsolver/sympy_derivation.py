"""Compatibility exports for symbolic derivation strategies.

New code should import from :mod:`stepsolver.derivation` modules by responsibility.
"""

from stepsolver.derivation.definite_integrals import (
    derive_definite_integral,
    derive_dirichlet_integral,
    derive_improper_integral,
)
from stepsolver.derivation.derivatives import derive_derivative
from stepsolver.derivation.equations import derive_polynomial_equation
from stepsolver.derivation.integrals_elementary import (
    derive_basic_antiderivative,
    derive_constant_multiple_integral,
    derive_function_substitution_integral,
    derive_log_derivative_integral,
    derive_polynomial_sum_integral,
)
from stepsolver.derivation.integrals_special import (
    derive_gaussian_antiderivative,
    derive_integration_by_parts,
    derive_inverse_hyperbolic_integral,
    derive_partial_fraction_integral,
    derive_shifted_semicircle_integral,
    derive_square_root_rational_integral,
    derive_trigonometric_power_integral,
)
from stepsolver.derivation.limits import derive_limit
from stepsolver.derivation.model import (
    BackendCrossedOut,
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendDifferential,
    BackendEvaluationAtBounds,
    BackendExpression,
    BackendIdentity,
    BackendIntegral,
    BackendIntegrationByPartsRule,
    BackendIntroduced,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendQuadraticSolutions,
    BackendQuotient,
    BackendSum,
    EquationBackendExpression,
)
from stepsolver.derivation.reciprocal_quadratic import derive_reciprocal_quadratic_integral

__all__ = [
    "BackendCrossedOut",
    "BackendDerivationStep",
    "BackendDerivative",
    "BackendDifference",
    "BackendDifferential",
    "BackendEvaluationAtBounds",
    "BackendExpression",
    "BackendIdentity",
    "BackendIntegral",
    "BackendIntegrationByPartsRule",
    "BackendIntroduced",
    "BackendLimit",
    "BackendMathNote",
    "BackendNotEqual",
    "BackendProduct",
    "BackendQuadraticSolutions",
    "BackendQuotient",
    "BackendSum",
    "EquationBackendExpression",
    "derive_basic_antiderivative",
    "derive_constant_multiple_integral",
    "derive_definite_integral",
    "derive_derivative",
    "derive_dirichlet_integral",
    "derive_function_substitution_integral",
    "derive_gaussian_antiderivative",
    "derive_improper_integral",
    "derive_integration_by_parts",
    "derive_inverse_hyperbolic_integral",
    "derive_limit",
    "derive_log_derivative_integral",
    "derive_partial_fraction_integral",
    "derive_polynomial_equation",
    "derive_polynomial_sum_integral",
    "derive_reciprocal_quadratic_integral",
    "derive_shifted_semicircle_integral",
    "derive_square_root_rational_integral",
    "derive_trigonometric_power_integral",
]
