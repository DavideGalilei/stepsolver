"""Conversion between StepSolver values and SymPy objects."""

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import assert_never

import sympy as sp

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Constant,
    ConstantName,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    OpaqueExpression,
    Operation,
    Relation,
    RelationOperator,
    SequenceExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.errors import BackendError, ParseError, QueryError
from stepsolver.parser import parse_expression
from stepsolver.results import (
    BooleanValue,
    MappingEntry,
    MappingValue,
    MathValue,
    MatrixValue,
    ScalarValue,
    SequenceValue,
)
from stepsolver.sympy_support import (
    expect_integer,
    expect_symbol,
    is_object_mapping,
    is_object_sequence,
)

_INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")
_RATIONAL_PATTERN = re.compile(r"^-?[0-9]+/[0-9]+$")
_DECIMAL_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]+$")


class SympyConverter:
    """Translate between the public mathematical model and backend-native values."""

    def to_sympy(self, expression: Expression) -> sp.Basic:
        """Convert one public AST expression to its SymPy equivalent."""
        if isinstance(expression, Number):
            return sp.Rational(expression.value.numerator, expression.value.denominator)
        if isinstance(expression, ApproximateNumber):
            return sp.Float(expression.text)
        if isinstance(expression, Symbol):
            return sp.Symbol(expression.name, real=True)
        if isinstance(expression, Constant):
            constants: dict[ConstantName, sp.Expr] = {
                ConstantName.PI: sp.pi,
                ConstantName.E: sp.E,
                ConstantName.IMAGINARY: sp.I,
                ConstantName.INFINITY: sp.oo,
            }
            return constants[expression.name]
        if isinstance(expression, UnaryExpression):
            operand = self.to_sympy(expression.operand)
            if expression.operator is UnaryOperator.POSITIVE:
                return operand
            if expression.operator is UnaryOperator.NEGATIVE:
                return -operand
            if expression.operator is UnaryOperator.FACTORIAL:
                return sp.factorial(operand)
            assert_never(expression.operator)
        if isinstance(expression, BinaryExpression):
            left = self.to_sympy(expression.left)
            right = self.to_sympy(expression.right)
            if expression.operator is BinaryOperator.ADD:
                return left + right
            if expression.operator is BinaryOperator.SUBTRACT:
                return left - right
            if expression.operator is BinaryOperator.MULTIPLY:
                return left * right
            if expression.operator is BinaryOperator.DIVIDE:
                return left / right
            if expression.operator is BinaryOperator.POWER:
                return left**right
            assert_never(expression.operator)
        if isinstance(expression, FunctionCall):
            return self.function_to_sympy(expression)
        if isinstance(expression, Relation):
            left = self.to_sympy(expression.left)
            right = self.to_sympy(expression.right)
            if expression.operator is RelationOperator.EQUAL:
                return sp.Eq(left, right, evaluate=False)
            if expression.operator is RelationOperator.NOT_EQUAL:
                return sp.Ne(left, right, evaluate=False)
            relations = {
                RelationOperator.LESS: sp.Lt,
                RelationOperator.LESS_EQUAL: sp.Le,
                RelationOperator.GREATER: sp.Gt,
                RelationOperator.GREATER_EQUAL: sp.Ge,
            }
            return relations[expression.operator](left, right)
        if isinstance(expression, SequenceExpression):
            raise BackendError("a sequence is only valid where an operation expects one")
        raise BackendError("opaque expressions cannot be sent back to the symbolic backend")

    def function_to_sympy(self, expression: FunctionCall) -> sp.Basic:
        """Convert a public function call to a supported SymPy function."""
        arguments = tuple(self.to_sympy(argument) for argument in expression.arguments)
        name = str(expression.name)
        unary_functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "asinh": sp.asinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "factorial": sp.factorial,
            "gamma": sp.gamma,
        }
        unary = unary_functions.get(name)
        if unary is not None:
            if len(arguments) != 1:
                raise QueryError(f"{name} expects one argument")
            return unary(arguments[0])
        if name == "log":
            if len(arguments) not in {1, 2}:
                raise QueryError("log expects one or two arguments")
            if len(arguments) == 1:
                return sp.log(arguments[0])
            return sp.log(arguments[0], arguments[1])
        if name == Operation.DIFFERENTIATE.value:
            if len(arguments) not in {2, 3}:
                raise QueryError("nested diff expects two or three arguments")
            order = 1
            if len(expression.arguments) == 3:
                order = expect_integer(expression.arguments[2], role="derivative order")
            return sp.diff(arguments[0], arguments[1], order)
        return sp.Function(name)(*arguments)

    def equations(self, expression: Expression) -> sp.Basic | Sequence[sp.Basic]:
        """Convert one equation or a sequence of equations."""
        if isinstance(expression, SequenceExpression):
            return tuple(self.to_sympy(item) for item in expression.items)
        return self.to_sympy(expression)

    def symbols(self, expression: Expression) -> tuple[sp.Basic, ...]:
        """Convert one symbol or a sequence of solution symbols."""
        if isinstance(expression, SequenceExpression):
            return tuple(
                self.to_sympy(expect_symbol(item, role="solution variable"))
                for item in expression.items
            )
        return (self.to_sympy(expect_symbol(expression, role="solution variable")),)

    def matrix_from_expression(self, expression: Expression) -> sp.MatrixBase:
        """Build a rectangular SymPy matrix from nested public sequences."""
        matrix_expression = expression
        if isinstance(expression, FunctionCall) and expression.name == "matrix":
            if len(expression.arguments) != 1:
                raise QueryError("matrix expects one nested sequence")
            matrix_expression = expression.arguments[0]
        if not isinstance(matrix_expression, SequenceExpression):
            raise QueryError("matrix data must be a nested sequence")
        rows: list[list[sp.Basic]] = []
        width: int | None = None
        for row_expression in matrix_expression.items:
            if not isinstance(row_expression, SequenceExpression):
                raise QueryError("each matrix row must be a sequence")
            row = [self.to_sympy(item) for item in row_expression.items]
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise QueryError("matrix rows must have equal length")
            rows.append(row)
        if not rows or width == 0:
            raise QueryError("matrix cannot be empty")
        return sp.Matrix(rows)

    def to_value(self, value: object) -> MathValue:
        """Convert a backend result to a typed public mathematical value."""
        if isinstance(value, bool):
            return BooleanValue(value=value)
        if isinstance(value, int):
            return ScalarValue(expression=Number(value=Fraction(value)))
        if isinstance(value, sp.MatrixBase):
            rows = tuple(
                tuple(self.from_sympy(value[row, column]) for column in range(value.cols))
                for row in range(value.rows)
            )
            return MatrixValue(rows=rows)
        if isinstance(value, sp.Basic):
            return ScalarValue(expression=self.from_sympy(value))
        if is_object_mapping(value):
            entries: list[MappingEntry] = []
            for key, item in value.items():
                key_expression = self.object_to_expression(key)
                entries.append(MappingEntry(key=key_expression, value=self.to_value(item)))
            entries.sort(key=lambda entry: str(entry.key))
            return MappingValue(entries=tuple(entries))
        if is_object_sequence(value):
            return SequenceValue(items=tuple(self.to_value(item) for item in value))
        raise BackendError(f"unsupported backend result type: {type(value).__name__}")

    def object_to_expression(self, value: object) -> Expression:
        """Convert a backend mapping key to a public expression."""
        if isinstance(value, int):
            return Number(value=Fraction(value))
        if isinstance(value, sp.Basic):
            return self.from_sympy(value)
        raise BackendError(f"unsupported mapping key type: {type(value).__name__}")

    def from_sympy(self, value: sp.Basic) -> Expression:
        """Convert a SymPy scalar to the closest public AST expression."""
        integration_constant = sp.Symbol("C")
        if value.func == sp.Abs and len(value.args) == 1:
            return FunctionCall(
                name=Identifier("abs"),
                arguments=(self.from_sympy(value.args[0]),),
            )
        if value.func == sp.Add and value.has(integration_constant):
            nonconstant_terms = tuple(term for term in value.args if term != integration_constant)
            if nonconstant_terms and all(
                not term.has(integration_constant) for term in nonconstant_terms
            ):
                expression = self.from_sympy(nonconstant_terms[0])
                for term in nonconstant_terms[1:]:
                    expression = BinaryExpression(
                        operator=BinaryOperator.ADD,
                        left=expression,
                        right=self.from_sympy(term),
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

    def step_expression(self, value: object) -> Expression:
        """Convert a backend result to an expression suitable for a fallback step."""
        if isinstance(value, sp.Basic):
            return self.from_sympy(value)
        if isinstance(value, bool):
            return OpaqueExpression(text="true" if value else "false")
        if isinstance(value, int):
            return Number(value=Fraction(value))
        return OpaqueExpression(text=str(value).replace("**", "^"))
