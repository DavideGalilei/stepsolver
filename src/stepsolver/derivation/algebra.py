"""Student-facing rule selection for elementary algebra transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Constant,
    Expression,
    FunctionCall,
    Number,
    OpaqueExpression,
    Operation,
    Relation,
    SequenceExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True, kw_only=True)
class AlgebraStepDescription:
    """A named algebra transformation and its student-facing explanation."""

    rule: str
    explanation: str


def _walk(expression: Expression) -> Iterator[Expression]:
    yield expression
    if isinstance(expression, UnaryExpression):
        yield from _walk(expression.operand)
    elif isinstance(expression, BinaryExpression):
        yield from _walk(expression.left)
        yield from _walk(expression.right)
    elif isinstance(expression, FunctionCall):
        for argument in expression.arguments:
            yield from _walk(argument)
    elif isinstance(expression, Relation):
        yield from _walk(expression.left)
        yield from _walk(expression.right)
    elif isinstance(expression, SequenceExpression):
        for item in expression.items:
            yield from _walk(item)


def _contains_operator(expression: Expression, operator: BinaryOperator) -> bool:
    return any(
        isinstance(item, BinaryExpression) and item.operator is operator
        for item in _walk(expression)
    )


def _contains_product_of_sum(expression: Expression) -> bool:
    for item in _walk(expression):
        if not isinstance(item, BinaryExpression):
            continue
        if item.operator is BinaryOperator.MULTIPLY and (
            _is_additive(item.left) or _is_additive(item.right)
        ):
            return True
        if (
            item.operator is BinaryOperator.POWER
            and _is_additive(item.left)
            and isinstance(item.right, Number)
            and item.right.value.denominator == 1
            and item.right.value.numerator > 1
        ):
            return True
    return False


def _is_additive(expression: Expression) -> bool:
    return isinstance(expression, BinaryExpression) and expression.operator in {
        BinaryOperator.ADD,
        BinaryOperator.SUBTRACT,
    }


def _has_symbolic_denominator(expression: Expression) -> bool:
    return any(
        isinstance(item, BinaryExpression)
        and item.operator is BinaryOperator.DIVIDE
        and _contains_symbol(item.right)
        for item in _walk(expression)
    )


def _contains_symbol(expression: Expression) -> bool:
    return any(isinstance(item, Symbol) for item in _walk(expression))


def _contains_function(expression: Expression) -> bool:
    return any(isinstance(item, FunctionCall) for item in _walk(expression))


def _is_exact_numeric(expression: Expression) -> bool:
    return all(
        isinstance(item, Number | UnaryExpression | BinaryExpression)
        and not (
            isinstance(item, UnaryExpression) and item.operator is UnaryOperator.FACTORIAL
        )
        for item in _walk(expression)
    )


def _is_rational_sum(expression: Expression) -> bool:
    if not _is_additive(expression) or _contains_symbol(expression):
        return False
    divisions = [
        item
        for item in _walk(expression)
        if isinstance(item, BinaryExpression) and item.operator is BinaryOperator.DIVIDE
    ]
    return bool(divisions) and all(
        isinstance(item.left, Number) and isinstance(item.right, Number) for item in divisions
    )


def _has_repeated_power_bases(expression: Expression) -> bool:
    """Recognize products where the exponent-addition law is the main idea."""
    for item in _walk(expression):
        if not isinstance(item, BinaryExpression) or item.operator is not BinaryOperator.MULTIPLY:
            continue
        factors = tuple(_multiplicative_factors(item))
        bases = tuple(_power_base(factor) for factor in factors)
        for index, base in enumerate(bases):
            if any(base == other for other in bases[index + 1 :]):
                return True
    return False


def _multiplicative_factors(expression: Expression) -> Iterator[Expression]:
    if isinstance(expression, BinaryExpression) and expression.operator is BinaryOperator.MULTIPLY:
        yield from _multiplicative_factors(expression.left)
        yield from _multiplicative_factors(expression.right)
        return
    yield expression


def _power_base(expression: Expression) -> Expression:
    if isinstance(expression, BinaryExpression) and expression.operator is BinaryOperator.POWER:
        return expression.left
    return expression


def _has_like_term_sum(expression: Expression) -> bool:
    if not _contains_operator(expression, BinaryOperator.ADD) and not _contains_operator(
        expression, BinaryOperator.SUBTRACT
    ):
        return False
    return _contains_symbol(expression)


def describe_algebra_step(
    operation: Operation,
    expression: Expression,
    backend_expression: sp.Basic,
    result: sp.Basic,
) -> AlgebraStepDescription:
    """Choose the most recognizable classroom rule for an algebra result."""
    operation_description = _operation_description(operation, expression)
    if operation_description is not None:
        return operation_description
    return _simplification_description(expression, backend_expression, result)


def _operation_description(
    operation: Operation,
    expression: Expression,
) -> AlgebraStepDescription | None:
    description: AlgebraStepDescription | None
    match operation:
        case Operation.EXPAND if _contains_product_of_sum(expression):
            description = AlgebraStepDescription(
                rule="Apply the distributive property",
                explanation=(
                    "Multiply each term across the parentheses, then combine any like terms."
                ),
            )
        case Operation.EXPAND:
            description = AlgebraStepDescription(
                rule="Expand the expression",
                explanation="Write products and positive integer powers as an expanded sum.",
            )
        case Operation.FACTOR:
            description = AlgebraStepDescription(
                rule="Factor the expression",
                explanation="Rewrite the expression as a product of simpler factors.",
            )
        case Operation.CANCEL:
            description = AlgebraStepDescription(
                rule="Cancel common factors",
                explanation=(
                    "Factor the numerator and denominator, then cancel only matching nonzero "
                    "factors."
                ),
            )
        case Operation.APART:
            description = AlgebraStepDescription(
                rule="Decompose into partial fractions",
                explanation="Rewrite the rational expression as a sum of simpler fractions.",
            )
        case _:
            description = None
    return description


def _simplification_description(
    expression: Expression,
    backend_expression: sp.Basic,
    result: sp.Basic,
) -> AlgebraStepDescription:
    if _is_rational_sum(expression):
        description = AlgebraStepDescription(
            rule="Use a common denominator",
            explanation="Rewrite the fractions over a common denominator, combine, and reduce.",
        )
    elif _is_exact_numeric(expression):
        description = AlgebraStepDescription(
            rule="Evaluate the arithmetic",
            explanation="Follow the order of operations and reduce the exact numerical result.",
        )
    elif _has_symbolic_denominator(expression) and sp.cancel(backend_expression) == result:
        description = AlgebraStepDescription(
            rule="Cancel common factors",
            explanation=(
                "Factor the numerator and denominator, then cancel matching factors while "
                "retaining the original denominator restrictions."
            ),
        )
    elif _contains_product_of_sum(expression):
        description = AlgebraStepDescription(
            rule="Expand and combine like terms",
            explanation=(
                "Use the distributive property, multiply the coefficients, and collect like terms."
            ),
        )
    elif _contains_function(expression):
        description = AlgebraStepDescription(
            rule="Apply algebraic identities",
            explanation="Use the applicable function identities, then simplify the arithmetic.",
        )
    elif _has_like_term_sum(expression):
        description = AlgebraStepDescription(
            rule="Combine like terms",
            explanation=(
                "Simplify each product, group terms with the same variable part, and add their "
                "coefficients."
            ),
        )
    elif _has_repeated_power_bases(expression):
        description = AlgebraStepDescription(
            rule="Combine powers with the same base",
            explanation=(
                "When multiplying powers with the same base, add their exponents and multiply "
                "the numerical coefficients."
            ),
        )
    else:
        description = AlgebraStepDescription(
            rule="Apply algebraic identities",
            explanation="Use the applicable sign, power, and arithmetic identities to simplify.",
        )
    return description


def symbolic_denominators(expression: Expression) -> tuple[Expression, ...]:
    """Return distinct input denominators that contain variables."""
    denominators: list[Expression] = []
    for item in _walk(expression):
        if (
            isinstance(item, BinaryExpression)
            and item.operator is BinaryOperator.DIVIDE
            and _contains_symbol(item.right)
            and item.right not in denominators
        ):
            denominators.append(item.right)
    return tuple(denominators)


def is_algebra_scalar(expression: object) -> bool:
    """Narrow helper used by strict type checkers at the operation boundary."""
    return isinstance(
        expression,
        Number
        | ApproximateNumber
        | Symbol
        | Constant
        | UnaryExpression
        | BinaryExpression
        | FunctionCall
        | OpaqueExpression,
    )
