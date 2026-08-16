"""Human-readable differentiation strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

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
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Callable

_BINARY_ARITY = 2
_MINIMUM_PRODUCT_FACTORS = 2


def _verified_derivative_steps(
    *,
    rule: str,
    expression: sp.Basic,
    variable: sp.Symbol,
    raw_derivative: sp.Basic,
    result: sp.Basic,
    explanation: str,
    notes: tuple[BackendMathNote, ...],
    show_simplification: bool = False,
) -> tuple[BackendDerivationStep, ...]:
    if str(sp.simplify(raw_derivative - result)) != "0":
        message = "the derivative rule did not match the exact backend result"
        raise ValueError(message)
    steps = [
        BackendDerivationStep(
            rule=rule,
            before=BackendDerivative(expression=expression, variable=variable),
            after=raw_derivative,
            explanation=explanation,
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "The displayed rule was applied symbolically and checked against the exact "
                "derivative."
            ),
            notes=notes,
        )
    ]
    if show_simplification and str(raw_derivative) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Simplify the derivative",
                before=raw_derivative,
                after=result,
                explanation="Combine factors and like terms into a cleaner final form.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between both derivative forms gives zero."
                ),
            )
        )
    return tuple(steps)


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


def _derive_quotient(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
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
    return _verified_derivative_steps(
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


def _derive_product(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
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
    return _verified_derivative_steps(
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


def _derive_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    terms = tuple(expression.as_ordered_terms())
    if len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    derivatives = tuple(sp.diff(term, variable) for term in terms)
    raw_derivative = sp.Add(*derivatives, evaluate=False)
    generic_f = sp.Function("f")(variable)
    generic_g = sp.Function("g")(variable)
    return _verified_derivative_steps(
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


def _derive_monomial(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not expression.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(expression, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if str(sp.simplify(expression - coefficient * variable**degree)) != "0":
        return ()
    raw_derivative = coefficient * degree * variable ** (degree - 1)
    pattern_variable = sp.Symbol("t", real=True)
    pattern_exponent = sp.Symbol("n", real=True)
    return _verified_derivative_steps(
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


def _negative_sine(argument: sp.Basic) -> sp.Basic:
    return -sp.sin(argument)


def _reciprocal(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(argument, -1, evaluate=False)


def _derive_function_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    candidates: tuple[
        tuple[Callable[[sp.Basic], sp.Basic], Callable[[sp.Basic], sp.Basic], str],
        ...,
    ] = (
        (sp.sin, sp.cos, "Differentiate the sine"),
        (sp.cos, _negative_sine, "Differentiate the cosine"),
        (sp.exp, sp.exp, "Differentiate the exponential"),
        (sp.log, _reciprocal, "Differentiate the logarithm"),
    )
    for function, outer_derivative, direct_rule in candidates:
        if expression.func != function or len(expression.args) != 1:
            continue
        argument = expression.args[0]
        inner_derivative = sp.diff(argument, variable)
        outer_value = outer_derivative(argument)
        raw_derivative = (
            outer_value
            if inner_derivative == sp.Integer(1)
            else sp.Mul(outer_value, inner_derivative, evaluate=False)
        )
        is_direct = argument == variable
        notes: tuple[BackendMathNote, ...] = (
            BackendMathNote(
                label="Outer derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=function(sp.Symbol("u", real=True)),
                        variable=sp.Symbol("u", real=True),
                    ),
                    right=outer_derivative(sp.Symbol("u", real=True)),
                ),
            ),
        )
        if not is_direct:
            notes = (
                *notes,
                BackendMathNote(
                    label="Inner derivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(expression=argument, variable=variable),
                        right=inner_derivative,
                    ),
                ),
            )
        return _verified_derivative_steps(
            rule=direct_rule if is_direct else "Apply the chain rule",
            expression=expression,
            variable=variable,
            raw_derivative=raw_derivative,
            result=result,
            explanation=(
                "Use the basic derivative pair."
                if is_direct
                else "Differentiate the outer function, then multiply by the inner derivative."
            ),
            notes=notes,
        )
    return ()


def _derive_power_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not expression.is_Pow or len(expression.args) != _BINARY_ARITY:
        return ()
    base, exponent = expression.args
    if exponent.has(variable):
        return ()
    base_derivative = sp.diff(base, variable)
    outer_power_derivative = sp.Mul(
        exponent,
        sp.Pow(base, exponent - 1, evaluate=False),
        evaluate=False,
    )
    raw_derivative = (
        outer_power_derivative
        if base_derivative == sp.Integer(1)
        else sp.Mul(outer_power_derivative, base_derivative, evaluate=False)
    )
    return _verified_derivative_steps(
        rule="Apply the power and chain rules",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=("Differentiate the outer power, then multiply by the derivative of its base."),
        notes=(
            BackendMathNote(
                label="Base derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=base, variable=variable),
                    right=base_derivative,
                ),
            ),
        ),
    )


def derive_derivative(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a first derivative using the most specific familiar rule."""
    strategies = (
        _derive_quotient,
        _derive_monomial,
        _derive_sum,
        _derive_product,
        _derive_power_chain,
        _derive_function_chain,
    )
    for strategy in strategies:
        derivation = strategy(expression, variable, result)
        if derivation:
            return derivation
    return ()
