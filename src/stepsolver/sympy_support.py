"""Shared validation and structural helpers for the SymPy adapter."""

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
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
_AND = cast("Callable[[sp.Basic, sp.Basic], sp.Basic]", vars(sp)["And"])


def positive_condition(expression: sp.Basic) -> object:
    """Return the real-positivity assumption for an expression."""
    return _Q.positive(expression)


def negative_condition(expression: sp.Basic) -> object:
    """Return the real-negativity assumption for an expression."""
    return _Q.negative(expression)


def refine_expression(expression: sp.Basic, condition: object) -> sp.Basic:
    """Apply SymPy's untyped assumption refinement at a typed boundary."""
    return _REFINE(expression, condition)


_MAX_REAL_BOUNDARIES = 24


def _real_zero_boundaries(
    expressions: Sequence[sp.Basic],
    variable: sp.Symbol,
) -> tuple[sp.Basic, ...] | None:
    """Return the sorted real zeros contributed by the supplied expressions."""
    zeros: list[sp.Basic] = []
    for expression in expressions:
        if not expression.has(variable):
            continue
        try:
            solutions = cast("list[object]", sp.solve(sp.Eq(expression, 0), variable))
        except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
            return None
        for solution in solutions:
            if not isinstance(solution, sp.Basic) or solution.free_symbols:
                return None
            if solution.is_real is False:
                continue
            if solution not in zeros:
                zeros.append(solution)
    if len(zeros) > _MAX_REAL_BOUNDARIES:
        return None
    try:
        return tuple(sorted(zeros, key=lambda item: float(cast("sp.Float", sp.N(item)))))
    except (TypeError, ValueError):
        return None


def _interval_condition(
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> object:
    """Describe one open real interval as a SymPy assumption."""
    if lower == -sp.oo:
        return sp.Lt(variable, upper)
    if upper == sp.oo:
        return sp.Gt(variable, lower)
    return _AND(sp.Gt(variable, lower), sp.Lt(variable, upper))


def differentiates_on_real_intervals(
    antiderivative: sp.Basic,
    integrand: sp.Basic,
    variable: sp.Symbol,
) -> bool:
    """Check the derivative on every real interval separated by the integrand's poles.

    Antiderivatives that use ``log(abs(u))`` agree with the integrand only on
    intervals where ``u`` keeps a fixed sign.  Testing the difference on each
    interval formed by all real zeros of the denominator and of the absolute
    values exercises every sign combination without guessing one branch.
    """
    if variable.is_real is not True:
        return False
    difference = sp.diff(antiderivative, variable) - integrand
    candidates = [sp.fraction(sp.together(integrand))[1]]
    candidates.extend(
        absolute.args[0] for absolute in antiderivative.atoms(sp.Abs) if absolute.has(variable)
    )
    boundaries = _real_zero_boundaries(candidates, variable)
    if boundaries is None:
        return False
    edges = (-sp.oo, *boundaries, sp.oo)
    for lower, upper in pairwise(edges):
        condition = _interval_condition(variable, lower, upper)
        try:
            refined = refine_expression(difference, condition)
            if sp.simplify(refined) != sp.Integer(0):
                return False
        except (sp.PolynomialError, NotImplementedError, TypeError, ValueError):
            return False
    return True


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
