"""Parser and AST regression tests."""

from fractions import Fraction

import pytest

from stepsolver import ParseError, format_expression, parse, parse_expression
from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Number,
    Operation,
    Relation,
    SequenceExpression,
    UnaryExpression,
    UnaryOperator,
)


def test_bare_expression_becomes_simplification_query() -> None:
    """Bare arithmetic should use the default simplification operation."""
    query = parse("2 + 3 * 4")
    assert query.operation is Operation.SIMPLIFY
    assert len(query.arguments) == 1


def test_power_is_right_associative() -> None:
    """Exponentiation should associate from the right."""
    expression = parse_expression("2^3^2")
    assert isinstance(expression, BinaryExpression)
    assert expression.operator is BinaryOperator.POWER
    assert isinstance(expression.right, BinaryExpression)
    assert expression.right.operator is BinaryOperator.POWER


def test_unary_minus_has_lower_precedence_than_power() -> None:
    """A leading minus should apply after exponentiation."""
    expression = parse_expression("-2^2")
    assert isinstance(expression, UnaryExpression)
    assert expression.operator is UnaryOperator.NEGATIVE
    assert isinstance(expression.operand, BinaryExpression)


def test_decimal_is_stored_exactly() -> None:
    """Decimal tokens should become exact rational values."""
    expression = parse_expression("0.125")
    assert expression == Number(value=Fraction(1, 8))


def test_relations_and_nested_sequences_parse() -> None:
    """Systems and matrices should retain their nested AST structure."""
    query = parse("solve([x+y=2, x-y=0], [x,y])")
    assert query.operation is Operation.SOLVE
    assert isinstance(query.arguments[0], SequenceExpression)
    equations = query.arguments[0]
    assert all(isinstance(item, Relation) for item in equations.items)
    assert isinstance(query.arguments[1], SequenceExpression)


@pytest.mark.parametrize(
    "source",
    ["", "2x", "__import__('os')", "x.y", "sqrt(2", "[1,2", "π"],
)
def test_invalid_or_non_ascii_input_is_rejected(source: str) -> None:
    """The parser should reject implicit multiplication and Python-like syntax."""
    with pytest.raises(ParseError):
        parse(source)


def test_web_parser_accepts_visual_editor_implicit_multiplication() -> None:
    """The opt-in web mode should normalize MathLive-style adjacent factors."""
    expression = parse_expression("2 x + 3(x+1)", allow_implicit_multiplication=True)
    assert format_expression(expression) == "2 * x + 3 * (x + 1)"


def test_formatted_integer_additions_round_trip() -> None:
    """Formatting and reparsing generated arithmetic should preserve its AST."""
    examples = (-1000, -17, -1, 0, 1, 23, 1000)
    for left in examples:
        for right in examples:
            expression = BinaryExpression(
                operator=BinaryOperator.ADD,
                left=Number(value=Fraction(left)),
                right=Number(value=Fraction(right)),
            )
            assert parse_expression(format_expression(expression)) == expression
