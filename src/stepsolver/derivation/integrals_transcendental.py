"""Trigonometric powers, Gaussian and inverse-hyperbolic integrals."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
    BackendProduct,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2


def derive_trigonometric_power_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Reduce an even square of sine or cosine before integrating."""
    candidates = (
        (
            sp.sin(variable) ** 2,
            (sp.Integer(1) - sp.cos(2 * variable)) / 2,
            "sine",
        ),
        (
            sp.cos(variable) ** 2,
            (sp.Integer(1) + sp.cos(2 * variable)) / 2,
            "cosine",
        ),
    )
    for candidate, reduced, name in candidates:
        if sp.simplify(integrand - candidate) != sp.Integer(0):
            continue
        integration_constant = sp.Symbol("C")
        antiderivative = sp.Add(sp.integrate(reduced, variable), integration_constant)
        if sp.simplify(antiderivative - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule=f"Use the {name} power-reduction identity",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=BackendIntegral(integrand=reduced, variable=variable),
                explanation=(
                    "Rewrite the squared trigonometric function using a double-angle identity."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The power-reduction identity preserves the integrand.",
                notes=(
                    BackendMathNote(
                        label="Power-reduction identity",
                        expression=BackendIdentity(left=candidate, right=reduced),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Integrate the reduced expression",
                before=BackendIntegral(integrand=reduced, variable=variable),
                after=result,
                explanation="Integrate the constant and double-angle cosine term separately.",
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the result recovers the original square.",
            ),
        )
    return ()


def derive_gaussian_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Explain the standard special-function antiderivative of the Gaussian."""
    gaussian = sp.exp(-(variable**2))
    if sp.simplify(integrand - gaussian) != sp.Integer(0):
        return ()
    integration_constant = sp.Symbol("C")
    formula = sp.sqrt(sp.pi) * sp.erf(variable) / 2 + integration_constant
    if sp.simplify(formula - result) != sp.Integer(0):
        return ()
    generic_variable = sp.Symbol("t", real=True)
    definition_variable = sp.Symbol("z", real=True)
    return (
        BackendDerivationStep(
            rule="Express the antiderivative with the error function",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=formula,
            explanation=(
                "The Gaussian has no elementary antiderivative, so use the standard error "
                "function and include the integration constant."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the error-function expression gives the Gaussian exactly."
            ),
            notes=(
                BackendMathNote(
                    label="Definition of the error function",
                    expression=BackendIdentity(
                        left=sp.erf(definition_variable),
                        right=BackendProduct(
                            factors=(
                                2 / sp.sqrt(sp.pi),
                                BackendIntegral(
                                    integrand=sp.exp(-(generic_variable**2)),
                                    variable=generic_variable,
                                    lower=sp.Integer(0),
                                    upper=definition_variable,
                                ),
                            )
                        ),
                    ),
                ),
            ),
        ),
    )


def derive_inverse_hyperbolic_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Normalize 1/sqrt(a*x^2+b) to the inverse-hyperbolic-sine rule."""
    radicand = sp.simplify(integrand**-2)
    try:
        polynomial = sp.Poly(radicand, variable)
    except sp.PolynomialError:
        return ()
    if polynomial.degree() != _QUADRATIC_DEGREE:
        return ()
    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    if (
        coefficient_b != sp.Integer(0)
        or coefficient_a.is_positive is not True
        or coefficient_c.is_positive is not True
        or sp.simplify(integrand - 1 / sp.sqrt(radicand)) != sp.Integer(0)
    ):
        return ()
    substitution_variable = sp.Symbol("u", real=True)
    scale = sp.sqrt(coefficient_a / coefficient_c)
    coefficient = 1 / sp.sqrt(coefficient_a)
    transformed_integrand = 1 / sp.sqrt(substitution_variable**2 + 1)
    transformed_integral = BackendIntegral(
        integrand=transformed_integrand,
        variable=substitution_variable,
        coefficient=coefficient,
    )
    integration_constant = sp.Symbol("C")
    transformed_formula = coefficient * sp.asinh(substitution_variable) + integration_constant
    final_formula = coefficient * sp.asinh(scale * variable) + integration_constant
    if sp.simplify(final_formula - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Normalize the quadratic radical",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "Scale the variable so the expression under the square root becomes one plus "
                "u squared."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The variable scaling produces the normalized radical exactly.",
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=scale * variable,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=substitution_variable,
                            coefficient=1 / scale,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the inverse hyperbolic sine rule",
            before=transformed_integral,
            after=transformed_formula,
            explanation=("The normalized integrand is the derivative of inverse hyperbolic sine."),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating inverse hyperbolic sine gives the normalized integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=transformed_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.asinh(substitution_variable) + integration_constant,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_formula,
            after=final_formula,
            explanation="Replace the temporary variable with its scaled x-expression.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Substitution gives the exact antiderivative in x.",
        ),
    )
