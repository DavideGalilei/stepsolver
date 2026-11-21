"""Integration by parts with a polynomial factor."""

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


def derive_power_log_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Integrate ``c*x**m*log(x)`` by parts for nonnegative integer m."""
    logarithm = sp.log(variable)
    if not integrand.has(logarithm):
        return ()
    polynomial_part = sp.simplify(integrand / logarithm)
    if not polynomial_part.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(polynomial_part, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if sp.simplify(polynomial_part - coefficient * variable**degree) != sp.Integer(0):
        return ()
    next_degree = degree + 1
    integrated_polynomial = coefficient * variable**next_degree / next_degree
    remaining_integrand = sp.simplify(integrated_polynomial / variable)
    integration_constant = sp.Symbol("C")
    formula = sp.Add(
        integrated_polynomial * logarithm,
        -sp.integrate(remaining_integrand, variable),
        integration_constant,
        evaluate=False,
    )
    if sp.simplify(formula - result) != sp.Integer(0):
        return ()
    parts_expression = BackendDifference(
        left=BackendProduct(factors=(integrated_polynomial, logarithm)),
        right=BackendIntegral(integrand=remaining_integrand, variable=variable),
    )
    displayed_formula = BackendSum(
        terms=(
            BackendDifference(
                left=BackendProduct(factors=(integrated_polynomial, logarithm)),
                right=sp.integrate(remaining_integrand, variable),
            ),
            integration_constant,
        ),
    )
    return (
        BackendDerivationStep(
            rule="Choose integration by parts",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=parts_expression,
            explanation=(
                "Let u = log(x) and integrate the power factor to obtain v, then apply "
                "integral u dv = uv - integral v du."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The integration-by-parts identity produces the reduced integral.",
            notes=(
                BackendMathNote(
                    label="Choose u",
                    expression=BackendIdentity(left=sp.Symbol("u"), right=logarithm),
                ),
                BackendMathNote(
                    label="Compute v",
                    expression=BackendIdentity(
                        left=sp.Symbol("v"),
                        right=integrated_polynomial,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the simpler remaining integral",
            before=parts_expression,
            after=displayed_formula,
            explanation="Apply the power rule to the simpler remaining integral and add C.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final expression recovers the integrand.",
        ),
    )


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
