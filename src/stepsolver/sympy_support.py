"""Shared validation and structural helpers for the SymPy adapter."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeGuard, cast

import sympy as sp

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    Constant,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    OpaqueExpression,
    Query,
    Relation,
    SequenceExpression,
    Symbol,
    UnaryExpression,
)
from stepsolver.errors import QueryError

_UNEVALUATED_OPERATION_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        ("sympy.concrete.products", "Product"),
        ("sympy.concrete.summations", "Sum"),
        ("sympy.core.function", "Derivative"),
        ("sympy.integrals.integrals", "Integral"),
        ("sympy.series.limits", "Limit"),
    }
)


class _AssumptionPredicates(Protocol):
    """The assumption predicates used by interval-aware verification."""

    def positive(self, expression: sp.Basic) -> object: ...

    def negative(self, expression: sp.Basic) -> object: ...


_Q = cast("_AssumptionPredicates", vars(sp)["Q"])
_REFINE = cast("Callable[[sp.Basic, object], sp.Basic]", vars(sp)["refine"])


def positive_condition(expression: sp.Basic) -> object:
    """Return the real-positivity assumption for an expression."""
    return _Q.positive(expression)


def negative_condition(expression: sp.Basic) -> object:
    """Return the real-negativity assumption for an expression."""
    return _Q.negative(expression)


def refine_expression(expression: sp.Basic, condition: object) -> sp.Basic:
    """Apply SymPy's untyped assumption refinement at a typed boundary."""
    return _REFINE(expression, condition)


def is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Narrow an arbitrary backend value to a mapping."""
    return isinstance(value, Mapping)


def is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow a non-string backend value to a sequence."""
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def is_real_expression(value: sp.Basic) -> bool:
    """Return whether a solved expression is compatible with a real-valued symbol."""
    if value.is_real is not None:
        return value.is_real
    if value.free_symbols:
        return True
    return sp.N(value).is_real is not False


def has_fractional_power(expression: sp.Basic) -> bool:
    """Return whether an expression contains a non-integer power.

    SymPy stores ``sqrt(x)`` as ``x**(1/2)``, so asking for ``sp.sqrt`` as a
    function atom never matches. Checking exponents keeps cheap structural
    guards honest for square roots and other radicals.
    """
    return any(power.args[1].is_integer is not True for power in expression.atoms(sp.Pow))


def contains_unevaluated_operation(value: object) -> bool:
    """Return whether a backend result still contains an unevaluated operation."""
    if isinstance(value, sp.Basic):
        operation_type = (type(value).__module__, type(value).__name__)
        return operation_type in _UNEVALUATED_OPERATION_TYPES or any(
            contains_unevaluated_operation(argument) for argument in value.args
        )
    if is_object_mapping(value):
        return any(
            contains_unevaluated_operation(key) or contains_unevaluated_operation(item)
            for key, item in value.items()
        )
    if is_object_sequence(value):
        return any(contains_unevaluated_operation(item) for item in value)
    return False


def query_expression(query: Query) -> FunctionCall:
    """Represent a query itself as a displayable function call."""
    return FunctionCall(name=Identifier(query.operation.value), arguments=query.arguments)


def expect_arity(query: Query, *allowed: int) -> None:
    """Require that a query has one of the allowed argument counts."""
    if len(query.arguments) not in allowed:
        choices = ", ".join(str(value) for value in allowed)
        raise QueryError(
            f"{query.operation.value} expects {choices} argument(s), got {len(query.arguments)}"
        )


def expect_symbol(expression: Expression, *, role: str) -> Symbol:
    """Require a symbol in a role-specific query position."""
    if not isinstance(expression, Symbol):
        raise QueryError(f"{role} must be a symbol")
    return expression


def expect_integer(expression: Expression, *, role: str) -> int:
    """Require and unwrap an integer-valued AST number."""
    if not isinstance(expression, Number) or expression.value.denominator != 1:
        raise QueryError(f"{role} must be an integer")
    return expression.value.numerator


def substitute(expression: Expression, symbol: Symbol, replacement: Expression) -> Expression:
    """Substitute one symbol structurally in the custom AST."""
    if isinstance(expression, Symbol):
        return replacement if expression == symbol else expression
    if isinstance(expression, Number | ApproximateNumber | Constant | OpaqueExpression):
        return expression
    if isinstance(expression, UnaryExpression):
        return UnaryExpression(
            operator=expression.operator,
            operand=substitute(expression.operand, symbol, replacement),
        )
    if isinstance(expression, BinaryExpression):
        return BinaryExpression(
            operator=expression.operator,
            left=substitute(expression.left, symbol, replacement),
            right=substitute(expression.right, symbol, replacement),
        )
    if isinstance(expression, FunctionCall):
        return FunctionCall(
            name=expression.name,
            arguments=tuple(
                substitute(argument, symbol, replacement) for argument in expression.arguments
            ),
        )
    if isinstance(expression, Relation):
        return Relation(
            operator=expression.operator,
            left=substitute(expression.left, symbol, replacement),
            right=substitute(expression.right, symbol, replacement),
        )
    return SequenceExpression(
        items=tuple(substitute(item, symbol, replacement) for item in expression.items)
    )
