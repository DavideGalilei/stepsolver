"""Radical rationalizations and shifted semicircle integrals."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2


def derive_square_root_rational_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Expose the hidden arctangent substitution in 1/(sqrt(x)(x+1))."""
    expected_integrand = 1 / (sp.sqrt(variable) * (variable + 1))
    if sp.simplify(integrand - expected_integrand) != sp.Integer(0):
        return ()
    substitution_variable = sp.Symbol("u", positive=True)
    transformed_integrand = 2 / (substitution_variable**2 + 1)
    transformed_integral = BackendIntegral(
        integrand=transformed_integrand,
        variable=substitution_variable,
    )
    integration_constant = sp.Symbol("C")
    transformed_result = 2 * sp.atan(substitution_variable) + integration_constant
    final_result = 2 * sp.atan(sp.sqrt(variable)) + integration_constant
    if sp.simplify(final_result - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Substitute the square root",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "The square root and its reciprocal suggest setting the new variable equal "
                "to the square root of x."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Replacing x and dx produces the displayed rational integral.",
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=sp.sqrt(variable),
                    ),
                ),
                BackendMathNote(
                    label="Rewrite x",
                    expression=BackendIdentity(
                        left=variable,
                        right=substitution_variable**2,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=substitution_variable,
                            coefficient=2 * substitution_variable,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the arctangent rule",
            before=transformed_integral,
            after=transformed_result,
            explanation="Apply the standard antiderivative of 1 divided by 1 plus u squared.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating gives the transformed integrand.",
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=1 / (substitution_variable**2 + 1),
                            variable=substitution_variable,
                        ),
                        right=sp.atan(substitution_variable) + integration_constant,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_result,
            after=final_result,
            explanation="Replace the temporary variable with the square root of x.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Direct substitution gives the final antiderivative.",
        ),
    )


def derive_shifted_semicircle_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Complete the square and use the standard semicircle antiderivative."""
    original_radicand = sp.simplify(integrand**2)
    expected_radicand = 2 * variable - variable**2
    if sp.simplify(original_radicand - expected_radicand) != sp.Integer(0):
        return ()
    shifted = variable - 1
    completed_radicand = 1 - shifted**2
    completed_integrand = sp.sqrt(completed_radicand)
    substitution_variable = sp.Symbol("u", real=True)
    transformed_integrand = sp.sqrt(1 - substitution_variable**2)
    integration_constant = sp.Symbol("C")
    transformed_formula = (
        substitution_variable * transformed_integrand / 2
        + sp.asin(substitution_variable) / 2
        + integration_constant
    )
    final_formula = shifted * completed_integrand / 2 + sp.asin(shifted) / 2 + integration_constant
    if sp.simplify(final_formula - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Complete the square under the radical",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=completed_integrand, variable=variable),
            explanation=(
                "Rewrite the quadratic as one minus a shifted square so it matches the "
                "standard semicircle form."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Expanding the completed square recovers the original radicand.",
            notes=(
                BackendMathNote(
                    label="Completed square",
                    expression=BackendIdentity(
                        left=expected_radicand,
                        right=completed_radicand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Shift the variable",
            before=BackendIntegral(integrand=completed_integrand, variable=variable),
            after=BackendIntegral(
                integrand=transformed_integrand,
                variable=substitution_variable,
            ),
            explanation="Set u equal to the shifted x-expression; its differential is dx.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The translation changes only the variable name.",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=shifted,
                    ),
                ),
                BackendMathNote(
                    label="Differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution_variable),
                        right=BackendDifferential(variable=variable),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the semicircle antiderivative",
            before=BackendIntegral(
                integrand=transformed_integrand,
                variable=substitution_variable,
            ),
            after=transformed_formula,
            explanation=("Apply the standard antiderivative for the upper unit semicircle."),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the formula gives the semicircle integrand.",
            notes=(
                BackendMathNote(
                    label="Standard rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=transformed_integrand,
                            variable=substitution_variable,
                        ),
                        right=transformed_formula,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_formula,
            after=final_formula,
            explanation="Replace u with x minus one.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Substitution returns the result to the original variable.",
        ),
    )
