"""Integration by parts and specialized antiderivative strategies."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendDifferential,
    BackendExpression,
    BackendIdentity,
    BackendIntegral,
    BackendIntegrationByPartsRule,
    BackendMathNote,
    BackendProduct,
    BackendSum,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2
_MINIMUM_PRODUCT_FACTORS = 2


def derive_integration_by_parts(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use integration by parts when differentiating a polynomial factor simplifies it."""
    polynomial_factors = tuple(
        factor
        for factor in integrand.as_ordered_factors()
        if factor.has(variable) and factor.is_polynomial(variable)
    )
    if len(polynomial_factors) != 1:
        return ()
    chosen_u = polynomial_factors[0]
    polynomial = sp.Poly(chosen_u, variable)
    if polynomial.degree() < 1:
        return ()
    chosen_dv = sp.simplify(integrand / chosen_u)
    if (
        chosen_dv.func not in {sp.exp, sp.sin, sp.cos}
        or len(chosen_dv.args) != 1
        or sp.diff(chosen_dv.args[0], variable).has(variable)
    ):
        return ()
    chosen_v = sp.integrate(chosen_dv, variable)
    if chosen_v.has(sp.Integral):
        return ()
    chosen_du = sp.diff(chosen_u, variable)
    remaining_integrand = sp.simplify(chosen_v * chosen_du)
    remaining_antiderivative = sp.integrate(remaining_integrand, variable)
    integration_constant = sp.Symbol("C")
    expected = sp.Add(
        chosen_u * chosen_v,
        -remaining_antiderivative,
        integration_constant,
        evaluate=False,
    )
    if remaining_antiderivative.has(sp.Integral) or sp.simplify(expected - result) != sp.Integer(0):
        return ()
    product_term = sp.simplify(chosen_u * chosen_v)
    if remaining_integrand.could_extract_minus_sign():
        by_parts_expression: BackendExpression = BackendSum(
            terms=(
                product_term,
                BackendIntegral(
                    integrand=-remaining_integrand,
                    variable=variable,
                ),
            )
        )
    else:
        by_parts_expression = BackendDifference(
            left=product_term,
            right=BackendIntegral(
                integrand=remaining_integrand,
                variable=variable,
            ),
        )
    return (
        BackendDerivationStep(
            rule="Choose integration by parts",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=by_parts_expression,
            explanation=(
                "Differentiate the polynomial factor because it becomes simpler, and "
                "antidifferentiate the other factor."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail=(
                "Expanding the integration-by-parts formula recovers the original integral."
            ),
            notes=(
                BackendMathNote(
                    label="Integration by parts",
                    expression=BackendIntegrationByPartsRule(),
                ),
                BackendMathNote(
                    label="Choose the algebraic part",
                    expression=BackendIdentity(
                        left=sp.Symbol("u"),
                        right=chosen_u,
                    ),
                ),
                BackendMathNote(
                    label="Choose the remaining differential",
                    expression=BackendIdentity(
                        left=sp.Symbol("dv"),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=(None if chosen_dv == sp.Integer(1) else chosen_dv),
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Differentiate u",
                    expression=BackendIdentity(
                        left=sp.Symbol("du"),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=(None if chosen_du == sp.Integer(1) else chosen_du),
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Antidifferentiate dv",
                    expression=BackendIdentity(
                        left=sp.Symbol("v"),
                        right=chosen_v,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the remaining integral",
            before=by_parts_expression,
            after=result,
            explanation=(
                "Integrate the simpler remaining term, combine the terms, and add the "
                "integration constant."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final expression recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Remaining antiderivative",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=remaining_integrand,
                            variable=variable,
                        ),
                        right=remaining_antiderivative,
                    ),
                ),
            ),
        ),
    )


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


def derive_partial_fraction_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Integrate rational functions whose partial fractions avoid logarithmic domains."""
    if not integrand.is_rational_function(variable) or result.has(sp.log):
        return ()
    decomposition = sp.apart(integrand, variable)
    terms = tuple(decomposition.as_ordered_terms())
    if str(decomposition) == str(integrand) or len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    if any(item.has(sp.Integral) for item in antiderivatives):
        return ()
    return (
        BackendDerivationStep(
            rule="Decompose into partial fractions",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=decomposition, variable=variable),
            explanation=(
                "Rewrite the rational function as a sum of simpler fractions that can be "
                "integrated separately."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the partial fractions recovers the original fraction.",
            notes=(
                BackendMathNote(
                    label="Partial-fraction identity",
                    expression=BackendIdentity(left=integrand, right=decomposition),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate each partial fraction",
            before=BackendIntegral(integrand=decomposition, variable=variable),
            after=result,
            explanation="Integrate the separated power and arctangent terms, then combine them.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final result recovers the rational function.",
            notes=tuple(
                BackendMathNote(
                    label=f"Term {index}",
                    expression=BackendIdentity(
                        left=BackendIntegral(integrand=term, variable=variable),
                        right=antiderivative,
                    ),
                )
                for index, (term, antiderivative) in enumerate(
                    zip(terms, antiderivatives, strict=True),
                    start=1,
                )
            ),
        ),
    )
