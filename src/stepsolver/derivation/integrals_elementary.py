"""Elementary antiderivative and substitution strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendSum,
)
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Callable

_QUADRATIC_DEGREE = 2
_MINIMUM_PRODUCT_FACTORS = 2


def _derive_power_log_integral(
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


def derive_log_derivative_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a logarithmic antiderivative using denominator substitution."""
    numerator, denominator = sp.fraction(sp.together(integrand))
    denominator_derivative = sp.diff(denominator, variable)
    if denominator_derivative == sp.Integer(0):
        return ()
    coefficient = sp.simplify(numerator / denominator_derivative)
    denominator_is_positive = denominator.is_positive is True
    if not denominator_is_positive and denominator.is_polynomial(variable):
        polynomial = sp.Poly(denominator, variable)
        if polynomial.degree() == _QUADRATIC_DEGREE:
            leading = polynomial.coeff_monomial(variable**2)
            linear = polynomial.coeff_monomial(variable)
            constant = polynomial.coeff_monomial(1)
            discriminant = sp.simplify(linear**2 - 4 * leading * constant)
            denominator_is_positive = (
                leading.is_positive is True and discriminant.is_negative is True
            )
    if coefficient.has(variable) or not denominator_is_positive:
        return ()
    substitution_variable = sp.Symbol("u", positive=True)
    unit_integrand = sp.Pow(substitution_variable, -1, evaluate=False)
    displayed_coefficient = None if coefficient == sp.Integer(1) else coefficient
    integration_constant = sp.Symbol("C")
    logarithm_in_substitution_variable = sp.log(substitution_variable)
    logarithm_in_original_variable = sp.log(denominator)
    formula_in_substitution_variable_term = (
        logarithm_in_substitution_variable
        if displayed_coefficient is None
        else sp.Mul(coefficient, logarithm_in_substitution_variable, evaluate=False)
    )
    formula_term = (
        logarithm_in_original_variable
        if displayed_coefficient is None
        else sp.Mul(coefficient, logarithm_in_original_variable, evaluate=False)
    )
    formula_in_substitution_variable = sp.Add(
        formula_in_substitution_variable_term,
        integration_constant,
        evaluate=False,
    )
    formula = sp.Add(formula_term, integration_constant, evaluate=False)
    if str(sp.simplify(sp.diff(formula, variable) - integrand)) != "0":
        message = "the logarithmic substitution failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the logarithmic antiderivative differs from the exact result"
        raise ValueError(message)
    transformed_integral = BackendIntegral(
        integrand=unit_integrand,
        variable=substitution_variable,
        coefficient=displayed_coefficient,
    )
    return (
        BackendDerivationStep(
            rule="Substitute the denominator",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "The numerator is a constant multiple of the denominator's derivative, so use "
                "the denominator as the new variable."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the denominator and its differential recovers the original integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=sp.Eq(
                        substitution_variable,
                        denominator,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Differentiate the substitution",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution_variable),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=denominator_derivative,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the logarithm rule",
            before=transformed_integral,
            after=formula_in_substitution_variable,
            explanation="The transformed integrand is the reciprocal function.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the logarithm recovers the reciprocal integrand."
            ),
            notes=(
                BackendMathNote(
                    label="The substitution is positive",
                    expression=sp.Gt(denominator, sp.Integer(0), evaluate=False),
                ),
                BackendMathNote(
                    label="Rule for positive inputs",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=unit_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.Add(
                            logarithm_in_substitution_variable,
                            integration_constant,
                            evaluate=False,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=formula_in_substitution_variable,
            after=formula,
            explanation="Replace the temporary variable with the original denominator.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Direct substitution gives an antiderivative equivalent to the exact result."
            ),
            notes=(
                BackendMathNote(
                    label="Replace the temporary variable",
                    expression=sp.Eq(
                        substitution_variable,
                        denominator,
                        evaluate=False,
                    ),
                ),
            ),
        ),
    )


def _negative_cosine(argument: sp.Basic) -> sp.Basic:
    return -sp.cos(argument)


def _derive_reverse_chain_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    candidates: tuple[
        tuple[Callable[[sp.Basic], sp.Basic], Callable[[sp.Basic], sp.Basic], str],
        ...,
    ] = (
        (sp.sin, _negative_cosine, "Apply the reverse chain rule for sine"),
        (sp.cos, sp.sin, "Apply the reverse chain rule for cosine"),
        (sp.exp, sp.exp, "Apply the reverse chain rule for the exponential"),
    )
    for function, outer_antiderivative, rule in candidates:
        if integrand.func != function or len(integrand.args) != 1:
            continue
        argument = integrand.args[0]
        inner_derivative = sp.simplify(sp.diff(argument, variable))
        if inner_derivative.has(variable) or inner_derivative == sp.Integer(0):
            return ()
        antiderivative = outer_antiderivative(argument) / inner_derivative
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if str(sp.simplify(formula - result)) != "0":
            return ()
        inner_variable = sp.Symbol("u", real=True)
        return (
            BackendDerivationStep(
                rule=rule,
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=formula,
                explanation=(
                    "Divide the outer antiderivative by the constant derivative of the inner "
                    "function."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "The chain rule confirms that differentiating the result recovers the "
                    "integrand."
                ),
                notes=(
                    BackendMathNote(
                        label="Inner function",
                        expression=sp.Eq(inner_variable, argument, evaluate=False),
                    ),
                    BackendMathNote(
                        label="Inner derivative",
                        expression=BackendIdentity(
                            left=BackendDerivative(expression=argument, variable=variable),
                            right=inner_derivative,
                        ),
                    ),
                    BackendMathNote(
                        label="Check with the chain rule",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=integrand,
                        ),
                    ),
                ),
            ),
        )
    return ()


def _derive_power_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    if not integrand.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(integrand, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if str(sp.simplify(integrand - coefficient * variable**degree)) != "0":
        return ()
    next_degree = degree + 1
    antiderivative = coefficient * variable**next_degree / next_degree
    formula = sp.Add(antiderivative, integration_constant, evaluate=False)
    if str(sp.simplify(formula - result)) != "0":
        return ()
    pattern_variable = sp.Symbol("t", real=True)
    pattern_exponent = sp.Symbol("n", real=True)
    pattern_integrand = sp.Pow(pattern_variable, pattern_exponent, evaluate=False)
    pattern_result = sp.Add(
        sp.Mul(
            sp.Pow(pattern_variable, pattern_exponent + 1, evaluate=False),
            sp.Pow(pattern_exponent + 1, -1, evaluate=False),
            evaluate=False,
        ),
        integration_constant,
        evaluate=False,
    )
    return (
        BackendDerivationStep(
            rule="Use the power rule",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=formula,
            explanation=(
                "Increase the exponent by one, divide by the new exponent, and keep any "
                "constant coefficient."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the power-rule result recovers the original monomial."
            ),
            notes=(
                BackendMathNote(
                    label="General power rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=pattern_integrand,
                            variable=pattern_variable,
                        ),
                        right=pattern_result,
                    ),
                ),
                BackendMathNote(
                    label="Restriction",
                    expression=BackendNotEqual(
                        left=pattern_exponent,
                        right=sp.Integer(-1),
                    ),
                ),
            ),
        ),
    )


def derive_basic_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive direct elementary antiderivatives from differentiation rules."""
    power_log = _derive_power_log_integral(integrand, variable, result)
    if power_log:
        return power_log
    integration_constant = sp.Symbol("C")
    candidates = (
        (sp.sin(variable), -sp.cos(variable), "Use the sine antiderivative"),
        (sp.cos(variable), sp.sin(variable), "Use the cosine antiderivative"),
        (sp.exp(variable), sp.exp(variable), "Use the exponential antiderivative"),
        (sp.sinh(variable), sp.cosh(variable), "Use the hyperbolic sine antiderivative"),
        (sp.cosh(variable), sp.sinh(variable), "Use the hyperbolic cosine antiderivative"),
        (
            sp.tan(variable),
            -sp.log(sp.cos(variable)),
            "Use the logarithmic derivative of cosine",
        ),
        (
            sp.Pow(1 - variable**2, sp.Rational(-1, 2)),
            sp.asin(variable),
            "Use the inverse-sine derivative pattern",
        ),
        (
            sp.Pow(1 + variable**2, sp.Rational(-1, 2)),
            sp.asinh(variable),
            "Use the inverse-hyperbolic-sine derivative pattern",
        ),
        (
            variable * sp.Pow(variable**2 + 1, sp.Rational(-1, 2)),
            sp.sqrt(variable**2 + 1),
            "Use the reverse chain rule for the square root",
        ),
    )
    for candidate_integrand, antiderivative, rule in candidates:
        if str(sp.simplify(integrand - candidate_integrand)) != "0":
            continue
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if str(sp.simplify(formula - result)) != "0":
            return ()
        return (
            BackendDerivationStep(
                rule=rule,
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=formula,
                explanation=(
                    "Use the matching derivative pair and include the integration constant."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "Differentiating the displayed antiderivative recovers the integrand."
                ),
                notes=(
                    BackendMathNote(
                        label="Derivative pair",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=candidate_integrand,
                        ),
                    ),
                ),
            ),
        )
    reverse_chain = _derive_reverse_chain_antiderivative(
        integrand,
        variable,
        result,
        integration_constant,
    )
    if reverse_chain:
        return reverse_chain
    return _derive_power_antiderivative(
        integrand,
        variable,
        result,
        integration_constant,
    )


def derive_polynomial_sum_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a polynomial integral term by term using linearity."""
    if not integrand.is_polynomial(variable):
        return ()
    terms = tuple(integrand.as_ordered_terms())
    if len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    integration_constant = sp.Symbol("C")
    formula = sp.Add(*antiderivatives, integration_constant, evaluate=False)
    if str(sp.simplify(formula - result)) != "0":
        return ()
    split_integral = BackendSum(
        terms=tuple(BackendIntegral(integrand=term, variable=variable) for term in terms)
    )
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    term_notes = tuple(
        BackendMathNote(
            label=f"Integrate term {index}",
            expression=BackendIdentity(
                left=BackendIntegral(integrand=term, variable=variable),
                right=antiderivative,
            ),
        )
        for index, (term, antiderivative) in enumerate(
            zip(terms, antiderivatives, strict=True),
            start=1,
        )
    )
    return (
        BackendDerivationStep(
            rule="Split the integral across the sum",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=split_integral,
            explanation="Use linearity to integrate each polynomial term separately.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Adding the separated integrands recovers the original polynomial.",
            notes=(
                BackendMathNote(
                    label="Linearity rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=sp.Add(function_f, function_g, evaluate=False),
                            variable=variable,
                        ),
                        right=BackendSum(
                            terms=(
                                BackendIntegral(integrand=function_f, variable=variable),
                                BackendIntegral(integrand=function_g, variable=variable),
                            )
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate each term",
            before=split_integral,
            after=formula,
            explanation="Apply the power rule to each term and add one integration constant.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the combined result recovers the polynomial.",
            notes=term_notes,
        ),
    )


def derive_constant_multiple_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive direct elementary integrals with a constant multiplier."""
    variable_factors = tuple(
        factor for factor in integrand.as_ordered_factors() if factor.has(variable)
    )
    if len(variable_factors) != 1:
        return ()
    variable_part = variable_factors[0]
    coefficient = sp.simplify(integrand / variable_part)
    if coefficient == sp.Integer(1) or coefficient.has(variable):
        return ()
    candidates = (
        (sp.sin(variable), -sp.cos(variable), "sine"),
        (sp.cos(variable), sp.sin(variable), "cosine"),
        (sp.exp(variable), sp.exp(variable), "exponential"),
    )
    for candidate, antiderivative, name in candidates:
        if str(sp.simplify(variable_part - candidate)) != "0":
            continue
        integration_constant = sp.Symbol("C")
        formula = sp.Add(
            sp.Mul(coefficient, antiderivative),
            integration_constant,
            evaluate=False,
        )
        if str(sp.simplify(formula - result)) != "0":
            return ()
        reduced_integral = BackendIntegral(
            integrand=variable_part,
            variable=variable,
            coefficient=coefficient,
        )
        return (
            BackendDerivationStep(
                rule="Factor out the constant",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=reduced_integral,
                explanation="Move the constant multiplier outside the integral.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Multiplying the reduced integrand by the constant recovers the original."
                ),
            ),
            BackendDerivationStep(
                rule=f"Use the {name} antiderivative",
                before=reduced_integral,
                after=formula,
                explanation="Apply the basic derivative pair, preserving the outside constant.",
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the result recovers the original integrand.",
                notes=(
                    BackendMathNote(
                        label="Derivative pair",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=candidate,
                        ),
                    ),
                ),
            ),
        )
    return ()


def derive_function_substitution_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a substitution when an inner derivative multiplies a basic function."""
    function_candidates = (
        (sp.sin, _negative_cosine),
        (sp.cos, sp.sin),
        (sp.exp, sp.exp),
    )
    for factor in integrand.as_ordered_factors():
        for function, outer_antiderivative in function_candidates:
            if factor.func != function or len(factor.args) != 1:
                continue
            argument = factor.args[0]
            argument_derivative = sp.diff(argument, variable)
            if argument_derivative == sp.Integer(0):
                continue
            coefficient = sp.simplify(integrand / (factor * argument_derivative))
            if coefficient.has(variable):
                continue
            substitution_variable = sp.Symbol("u", real=True)
            transformed_function = function(substitution_variable)
            transformed_integral = BackendIntegral(
                integrand=transformed_function,
                variable=substitution_variable,
                coefficient=None if coefficient == sp.Integer(1) else coefficient,
            )
            formula_in_substitution_variable_term = outer_antiderivative(substitution_variable)
            formula_term = outer_antiderivative(argument)
            if coefficient != sp.Integer(1):
                formula_in_substitution_variable_term = sp.Mul(
                    coefficient,
                    formula_in_substitution_variable_term,
                    evaluate=False,
                )
                formula_term = sp.Mul(coefficient, formula_term, evaluate=False)
            integration_constant = sp.Symbol("C")
            formula_in_substitution_variable = sp.Add(
                formula_in_substitution_variable_term,
                integration_constant,
                evaluate=False,
            )
            formula = sp.Add(formula_term, integration_constant, evaluate=False)
            if str(sp.simplify(formula - result)) != "0":
                return ()
            return (
                BackendDerivationStep(
                    rule="Substitute the inner function",
                    before=BackendIntegral(integrand=integrand, variable=variable),
                    after=transformed_integral,
                    explanation=(
                        "The derivative of the inner function appears as a factor, so use the "
                        "inner function as the new variable."
                    ),
                    verification_method=VerificationMethod.SUBSTITUTION,
                    verification_detail=(
                        "Replacing the inner function and differential recovers the original "
                        "integrand."
                    ),
                    notes=(
                        BackendMathNote(
                            label="Choose the substitution",
                            expression=sp.Eq(
                                substitution_variable,
                                argument,
                                evaluate=False,
                            ),
                        ),
                        BackendMathNote(
                            label="Differentiate the substitution",
                            expression=BackendIdentity(
                                left=BackendDifferential(variable=substitution_variable),
                                right=BackendDifferential(
                                    variable=variable,
                                    coefficient=argument_derivative,
                                ),
                            ),
                        ),
                    ),
                ),
                BackendDerivationStep(
                    rule="Integrate in the new variable",
                    before=transformed_integral,
                    after=formula_in_substitution_variable,
                    explanation="Apply the basic antiderivative in the substitution variable.",
                    verification_method=VerificationMethod.DIFFERENTIATION,
                    verification_detail=(
                        "Differentiating with respect to the new variable recovers the "
                        "transformed integrand."
                    ),
                ),
                BackendDerivationStep(
                    rule="Substitute back",
                    before=formula_in_substitution_variable,
                    after=formula,
                    explanation="Replace the temporary variable with the original inner function.",
                    verification_method=VerificationMethod.SUBSTITUTION,
                    verification_detail="Direct substitution gives the exact antiderivative.",
                ),
            )
    return ()
