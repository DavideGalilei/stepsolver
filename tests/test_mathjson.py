"""MathJSON notation decoding and automatic intent inference tests."""

from fractions import Fraction

import pytest

from stepsolver.ast import Number, Operation, Query, SequenceExpression, Symbol
from stepsolver.errors import QueryError
from stepsolver.formatter import format_expression
from stepsolver.mathjson import JsonValue, expression_from_mathjson, query_from_mathjson


@pytest.mark.parametrize(
    ("math_json", "operation", "argument_count"),
    [
        (["Add", "x", 2], Operation.SIMPLIFY, 1),
        (["Equal", ["Subtract", ["Power", "x", 2], 4], 0], Operation.SOLVE, 2),
        (["Less", "x", 4], Operation.SOLVE_INEQUALITY, 2),
        (["Integrate", ["Sin", "x"], "x"], Operation.INTEGRATE, 2),
        (
            ["Integrate", ["Sin", "x"], ["Tuple", "x", 0, "Pi"]],
            Operation.INTEGRATE,
            4,
        ),
        (["D", ["Sin", "x"], "x"], Operation.DIFFERENTIATE, 2),
        (["Derivative", ["Sin", "x"], "x"], Operation.DIFFERENTIATE, 2),
        (["D", ["D", ["Sin", "x"], "x"], "x"], Operation.DIFFERENTIATE, 3),
        (["Limit", ["Divide", ["Sin", "x"], "x"], "x", 0], Operation.LIMIT, 3),
        (
            ["Limit", ["Divide", ["Sin", "x"], "x"], ["Tuple", "x", 0]],
            Operation.LIMIT,
            3,
        ),
        (["Sum", "n", ["Tuple", "n", 1, 10]], Operation.SUM, 4),
        (["Product", "n", ["Tuple", "n", 1, 4]], Operation.PRODUCT, 4),
        (["List", ["List", 1, 2], ["List", 3, 4]], Operation.MATRIX, 1),
    ],
)
def test_query_inference(
    math_json: JsonValue,
    operation: Operation,
    argument_count: int,
) -> None:
    """Outer mathematical notation should select the solver operation."""
    query = query_from_mathjson(math_json)
    assert query.operation is operation
    assert len(query.arguments) == argument_count


def test_multivariable_equation_infers_a_variable_sequence() -> None:
    """Equations may infer all variables without a separate variable field."""
    query = query_from_mathjson(["Equal", ["Add", "x", "y"], 2])
    variables = query.arguments[1]
    assert isinstance(variables, SequenceExpression)
    assert all(isinstance(item, Symbol) for item in variables.items)


def test_equation_list_infers_a_system_solve() -> None:
    """Graphical cases notation should become one system query."""
    query = query_from_mathjson(
        [
            "List",
            ["Equal", ["Add", "x", "y"], 3],
            ["Equal", ["Subtract", "x", "y"], 1],
        ]
    )
    assert query.operation is Operation.SOLVE
    assert isinstance(query.arguments[0], SequenceExpression)
    assert isinstance(query.arguments[1], SequenceExpression)


def test_indefinite_integral_can_infer_its_variable() -> None:
    """A single-symbol integrand should not need an explicit differential."""
    query = query_from_mathjson(["Integrate", ["Power", "x", 2]])
    assert query.operation is Operation.INTEGRATE
    assert isinstance(query.arguments[1], Symbol)


@pytest.mark.parametrize(
    "math_json",
    [
        [],
        [1, "x"],
        None,
        True,
        {"sym": "x"},
        ["Error", "missing"],
        ["Add"],
        ["Hold", 1, 2],
        ["Divide", 1],
        ["Integrate", "x", "x", 0],
        ["Integrate", "x", ["List", "x", 0, 1]],
        ["Integrate", ["Add", "x", "y"]],
        ["D", "x"],
        ["D", ["D", "x", "x"], "y"],
        ["Limit", "x", "x"],
        ["Sum", "n", ["Tuple", "n", 1]],
        ["Equal", 1, 1],
    ],
)
def test_invalid_or_ambiguous_notation_is_rejected(math_json: JsonValue) -> None:
    """Incomplete or ambiguous visual input should produce a typed query error."""
    with pytest.raises(QueryError):
        query_from_mathjson(math_json)


def test_expression_decoder_handles_numeric_and_structural_forms() -> None:
    """Common Compute Engine nodes should map onto the closed custom AST."""
    square = expression_from_mathjson(["Square", ["Negate", 3]])
    product = expression_from_mathjson(["Multiply", 2, "x"])
    difference = expression_from_mathjson(["Subtract", "x", 2])
    rational = expression_from_mathjson(["Rational", 1, 3])
    custom = expression_from_mathjson(["CustomFunction", "x"])
    sequence = expression_from_mathjson(["Tuple", 1, 2.5])
    held = expression_from_mathjson(["Delimiter", "x"])
    assert square is not None
    assert product is not None
    assert difference is not None
    assert rational is not None
    assert custom is not None
    assert isinstance(sequence, SequenceExpression)
    assert isinstance(held, Symbol)
    assert expression_from_mathjson(2) == Number(value=Fraction(2))


def test_invisible_operator_decodes_as_implicit_multiplication() -> None:
    """Visual adjacency must become multiplication, not a backend function call."""
    visual_product: JsonValue = [
        "InvisibleOperator",
        ["Sin", "x"],
        ["Power", "ExponentialE", "x"],
    ]
    explicit_product: JsonValue = [
        "Multiply",
        ["Sin", "x"],
        ["Power", "ExponentialE", "x"],
    ]
    assert expression_from_mathjson(visual_product) == expression_from_mathjson(explicit_product)


def test_nested_calculus_operator_is_rejected_as_an_expression() -> None:
    """Calculus operators are semantic query roots, not ordinary functions."""
    with pytest.raises(QueryError, match="outermost"):
        expression_from_mathjson(["Add", 1, ["Integrate", "x", "x"]])


def test_indexed_root_mathjson_uses_the_supported_root_function() -> None:
    """MathLive indexed radicals should not become unknown backend functions."""
    expression = expression_from_mathjson(["Root", ["Power", 2, "n"], "n"])
    assert format_expression(expression) == "root(2 ^ n, n)"


def test_factorial_mathjson_uses_postfix_factorial_not_an_unknown_function() -> None:
    """MathLive factorial notation should remain a typed postfix operation."""
    expression = expression_from_mathjson(["Factorial", "n"])
    assert format_expression(expression) == "n!"


def test_mathjson_string_literal_is_rejected() -> None:
    """Compute Engine string values must not become mathematical symbols."""
    with pytest.raises(QueryError, match="string literals"):
        expression_from_mathjson("'text'")


def test_query_is_a_custom_ast_query() -> None:
    """The web notation boundary must return the library's public query abstraction."""
    assert isinstance(query_from_mathjson(["Add", 1, 2]), Query)
