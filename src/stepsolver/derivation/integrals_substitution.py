"""Logarithmic, linear-argument, and inner-function substitution."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.antiderivative_pairs import ELEMENTARY_PAIRS
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2


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
    if sp.simplify(sp.diff(formula, variable) - integrand) != sp.Integer(0):
        message = "the logarithmic substitution failed differentiation verification"
        raise ValueError(message)
    if sp.simplify(formula - result) != sp.Integer(0):
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


def derive_linear_argument_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    """Integrate an elementary function of a linear argument."""
    for pair in ELEMENTARY_PAIRS:
        if integrand.func != pair.function or len(integrand.args) != 1:
            continue
        argument = integrand.args[0]
        inner_derivative = sp.simplify(sp.diff(argument, variable))
        if inner_derivative.has(variable) or inner_derivative == sp.Integer(0):
            return ()
        antiderivative = pair.primitive(argument) / inner_derivative
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if sp.simplify(formula - result) != sp.Integer(0):
            return ()
        inner_variable = sp.Symbol("u", real=True)
        return (
            BackendDerivationStep(
                rule=pair.reverse_chain_rule,
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


def derive_function_substitution_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a substitution when an inner derivative multiplies a basic function."""
    for factor in integrand.as_ordered_factors():
        for pair in ELEMENTARY_PAIRS:
            if factor.func != pair.function or len(factor.args) != 1:
                continue
            argument = factor.args[0]
            argument_derivative = sp.diff(argument, variable)
            if argument_derivative == sp.Integer(0):
                continue
            coefficient = sp.simplify(integrand / (factor * argument_derivative))
            if coefficient.has(variable):
                continue
            substitution_variable = sp.Symbol("u", real=True)
            transformed_function = pair.function(substitution_variable)
            transformed_integral = BackendIntegral(
                integrand=transformed_function,
                variable=substitution_variable,
                coefficient=None if coefficient == sp.Integer(1) else coefficient,
            )
            formula_in_substitution_variable_term = pair.primitive(substitution_variable)
            formula_term = pair.primitive(argument)
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
            if sp.simplify(formula - result) != sp.Integer(0):
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
