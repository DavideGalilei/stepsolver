"""Differential checks for the AST-first SymPy conversion path."""

import re
from fractions import Fraction

import pytest
import sympy as sp

import stepsolver.sympy_conversion as conversion_module
from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    OpaqueExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.errors import ParseError
from stepsolver.formatter import format_expression
from stepsolver.parser import parse_expression
from stepsolver.sympy_conversion import SympyConverter

_INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")
_RATIONAL_PATTERN = re.compile(r"^-?[0-9]+/[0-9]+$")
_DECIMAL_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]+$")


def _legacy_from_sympy(  # ruff: ignore[too-many-return-statements]
    converter: SympyConverter,
    value: sp.Basic,
) -> Expression:
    """Recreate the text-based converter that preceded the fast AST path."""
    integration_constant = sp.Symbol("C")
    if value.func == sp.factorial and len(value.args) == 1:
        return UnaryExpression(
            operator=UnaryOperator.FACTORIAL,
            operand=_legacy_from_sympy(converter, value.args[0]),
        )
    if value.func == sp.Abs and len(value.args) == 1:
        return FunctionCall(
            name=Identifier("abs"),
            arguments=(_legacy_from_sympy(converter, value.args[0]),),
        )
    if value.func == sp.log and len(value.args) == 1:
        return FunctionCall(
            name=Identifier("log"),
            arguments=(_legacy_from_sympy(converter, value.args[0]),),
        )
    if value.func == sp.Add and value.has(integration_constant):
        nonconstant_terms = tuple(term for term in value.args if term != integration_constant)
        if nonconstant_terms and all(
            not term.has(integration_constant) for term in nonconstant_terms
        ):
            expression = _legacy_from_sympy(converter, nonconstant_terms[0])
            for term in nonconstant_terms[1:]:
                expression = BinaryExpression(
                    operator=BinaryOperator.ADD,
                    left=expression,
                    right=_legacy_from_sympy(converter, term),
                )
            return BinaryExpression(
                operator=BinaryOperator.ADD,
                left=expression,
                right=Symbol(name=Identifier("C")),
            )
    text = re.sub(r"\bI\b", "i", str(value)).replace("**", "^")
    if (
        _INTEGER_PATTERN.fullmatch(text) is not None
        or _RATIONAL_PATTERN.fullmatch(text) is not None
    ):
        return Number(value=Fraction(text))
    if _DECIMAL_PATTERN.fullmatch(text) is not None:
        return ApproximateNumber(text=text)
    try:
        return parse_expression(text)
    except ParseError:
        return OpaqueExpression(text=text)


@pytest.mark.parametrize(
    "value",
    [
        sp.Integer(3),
        sp.Integer(-3),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Float("1.25"),
        sp.Symbol("x"),
        sp.Symbol("C"),
        sp.Add(2 * sp.Symbol("x"), 5, evaluate=False),
        sp.Add(sp.Symbol("x"), 1, evaluate=False),
        sp.Add(sp.Symbol("x"), -1, evaluate=False),
        sp.Add(sp.Rational(1, 2), sp.Symbol("x"), evaluate=False),
        sp.Add(sp.Symbol("x") ** 3, sp.Symbol("C"), evaluate=False),
        sp.Mul(3, sp.Symbol("x") ** 2, evaluate=False),
        sp.Mul(2, sp.Symbol("x"), evaluate=False),
        sp.Mul(sp.Symbol("x"), sp.Symbol("y"), evaluate=False),
        sp.Mul(-1, sp.Symbol("x"), evaluate=False),
        sp.Mul(-2, sp.Symbol("x"), evaluate=False),
        sp.Mul(sp.Rational(1, 2), sp.Symbol("x"), evaluate=False),
        sp.Mul(2, sp.Symbol("x"), 1 / (sp.Symbol("x") ** 2 + 1), evaluate=False),
        sp.Pow(sp.Symbol("x"), 2),
        sp.Pow(sp.Symbol("x") + 1, 3),
        sp.Pow(sp.Symbol("x"), -1),
        sp.Pow(sp.Symbol("x"), sp.Rational(1, 2)),
        sp.sin(sp.Symbol("x")),
        sp.cos(sp.Symbol("x")),
        sp.exp(sp.Symbol("x")),
        sp.log(sp.Symbol("x")),
        sp.Abs(sp.Symbol("x")),
        sp.factorial(sp.Symbol("x")),
        sp.gamma(sp.Symbol("x")),
        sp.sin(sp.Symbol("x")) ** 2 + sp.cos(sp.Symbol("x")) ** 2,
        sp.exp(sp.Symbol("x")) * sp.sin(sp.Symbol("x")),
    ],
)
def test_ast_first_conversion_preserves_the_legacy_rendering(value: sp.Basic) -> None:
    """The fast path and the old text path must format identically."""
    converter = SympyConverter()
    fast = converter.from_sympy(value)
    legacy = _legacy_from_sympy(converter, value)
    assert format_expression(fast) == format_expression(legacy)


def test_simple_polynomials_do_not_reparse_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Straightforward integration output should be rendered from the AST once."""
    x = sp.Symbol("x", real=True)
    converter = SympyConverter()

    def fail_parse(_text: str) -> object:
        message = "simple expressions should not be reparsed"
        raise AssertionError(message)

    monkeypatch.setattr(conversion_module, "parse_expression", fail_parse)
    expression = converter.from_sympy(3 * x**2 + 2 * x + 5)
    assert format_expression(expression) == "3 * x ^ 2 + 2 * x + 5"
