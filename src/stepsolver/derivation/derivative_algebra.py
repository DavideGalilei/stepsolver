"""Product, quotient, sum, and monomial differentiation."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.derivative_steps import verified_derivative_steps
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendIdentity,
    BackendMathNote,
    BackendProduct,
    BackendQuotient,
    BackendSum,
)

_MINIMUM_PRODUCT_FACTORS = 2


def _generic_product_rule(variable: sp.Symbol) -> BackendIdentity:
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    return BackendIdentity(
        left=BackendDerivative(
            expression=sp.Mul(function_f, function_g, evaluate=False),
            variable=variable,
        ),
        right=BackendSum(
            terms=(
                BackendProduct(
                    factors=(
                        BackendDerivative(expression=function_f, variable=variable),
                        function_g,
                    )
                ),
                BackendProduct(
                    factors=(
                        function_f,
                        BackendDerivative(expression=function_g, variable=variable),
                    )
                ),
            )
        ),
    )


def _generic_quotient_rule(variable: sp.Symbol) -> BackendIdentity:
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    numerator = BackendDifference(
        left=BackendProduct(
            factors=(
                BackendDerivative(expression=function_f, variable=variable),
                function_g,
            )
        ),
        right=BackendProduct(
            factors=(
                function_f,
                BackendDerivative(expression=function_g, variable=variable),
            )
        ),
    )
    return BackendIdentity(
        left=BackendDerivative(
            expression=sp.Mul(function_f, sp.Pow(function_g, -1, evaluate=False), evaluate=False),
            variable=variable,
        ),
        right=BackendQuotient(
            numerator=numerator,
            denominator=sp.Pow(function_g, 2, evaluate=False),
        ),
    )


def derive_quotient(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate a quotient and retain its denominator condition."""
    numerator, denominator = sp.fraction(expression)
    if denominator == sp.Integer(1) or not denominator.has(variable):
        return ()
    numerator_derivative = sp.diff(numerator, variable)
    denominator_derivative = sp.diff(denominator, variable)
    first_term = sp.Mul(numerator_derivative, denominator)
    second_term = sp.Mul(numerator, denominator_derivative)
    raw_numerator = sp.Add(
        first_term,
        -second_term,
        evaluate=False,
    )
    raw_derivative = sp.Mul(
        raw_numerator,
        sp.Pow(denominator, -2, evaluate=False),
        evaluate=False,
    )
    return verified_derivative_steps(
        rule="Apply the quotient rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Differentiate the numerator and denominator separately, then apply the quotient "
            "rule in its standard order."
        ),
        notes=(
            BackendMathNote(label="Quotient rule", expression=_generic_quotient_rule(variable)),
            BackendMathNote(
                label="Numerator derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=numerator, variable=variable),
                    right=numerator_derivative,
                ),
            ),
            BackendMathNote(
                label="Denominator derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=denominator, variable=variable),
                    right=denominator_derivative,
                ),
            ),
        ),
        show_simplification=True,
    )


def derive_product(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate each factor in a product in turn."""
    variable_factors = tuple(
        factor for factor in expression.as_ordered_factors() if factor.has(variable)
    )
    if len(variable_factors) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    first_factor = variable_factors[0]
    second_factor = sp.Mul(*variable_factors[1:])
    constant = sp.simplify(expression / (first_factor * second_factor))
    product_derivative = sp.Add(
        sp.Mul(sp.diff(first_factor, variable), second_factor, evaluate=False),
        sp.Mul(first_factor, sp.diff(second_factor, variable), evaluate=False),
        evaluate=False,
    )
    raw_derivative = (
        product_derivative
        if constant == sp.Integer(1)
        else sp.Mul(constant, product_derivative, evaluate=False)
    )
    return verified_derivative_steps(
        rule="Apply the product rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation="Differentiate one factor at a time while leaving the other unchanged.",
        notes=(
            BackendMathNote(label="Product rule", expression=_generic_product_rule(variable)),
            BackendMathNote(
                label="First factor derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=first_factor, variable=variable),
                    right=sp.diff(first_factor, variable),
                ),
            ),
            BackendMathNote(
                label="Second factor derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=second_factor, variable=variable),
                    right=sp.diff(second_factor, variable),
                ),
            ),
        ),
        show_simplification=True,
    )


def derive_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate a sum term by term."""
    terms = tuple(expression.as_ordered_terms())
    if len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    derivatives = tuple(sp.diff(term, variable) for term in terms)
    raw_derivative = sp.Add(*derivatives, evaluate=False)
    generic_f = sp.Function("f")(variable)
    generic_g = sp.Function("g")(variable)
    return verified_derivative_steps(
        rule="Differentiate term by term",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation="Use linearity, then differentiate each term with its matching rule.",
        notes=(
            BackendMathNote(
                label="Sum rule",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=sp.Add(generic_f, generic_g, evaluate=False),
                        variable=variable,
                    ),
                    right=BackendSum(
                        terms=(
                            BackendDerivative(expression=generic_f, variable=variable),
                            BackendDerivative(expression=generic_g, variable=variable),
                        )
                    ),
                ),
            ),
            *tuple(
                BackendMathNote(
                    label=f"Term {index}",
                    expression=BackendIdentity(
                        left=BackendDerivative(expression=term, variable=variable),
                        right=derivative,
                    ),
                )
                for index, (term, derivative) in enumerate(
                    zip(terms, derivatives, strict=True),
                    start=1,
                )
            ),
        ),
    )


def derive_monomial(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Differentiate a power of the variable with a constant coefficient."""
    if not expression.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(expression, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if sp.simplify(expression - coefficient * variable**degree) != sp.Integer(0):
        return ()
    raw_derivative = coefficient * degree * variable ** (degree - 1)
    pattern_variable = sp.Symbol("t", real=True)
    pattern_exponent = sp.Symbol("n", real=True)
    return verified_derivative_steps(
        rule="Use the power rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Multiply by the exponent, then decrease the exponent by one; preserve the "
            "constant coefficient."
        ),
        notes=(
            BackendMathNote(
                label="General power rule",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=sp.Pow(pattern_variable, pattern_exponent, evaluate=False),
                        variable=pattern_variable,
                    ),
                    right=sp.Mul(
                        pattern_exponent,
                        sp.Pow(pattern_variable, pattern_exponent - 1, evaluate=False),
                        evaluate=False,
                    ),
                ),
            ),
        ),
    )
