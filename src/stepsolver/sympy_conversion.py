"""Conversion between StepSolver values and SymPy objects."""

import re
from collections.abc import Callable, Sequence
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

_UNARY_FUNCTIONS: dict[str, Callable[[sp.Basic], sp.Basic]] = {
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
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "factorial": sp.factorial,
    "gamma": sp.gamma,
}

_SUPPORTED_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        *_UNARY_FUNCTIONS,
        "log",
        "root",
        Operation.DIFFERENTIATE.value,
        "Derivative",
        "matrix",
    }
)

# Calculus operations may use an abstract function such as f(x).  Single-letter
# names keep that support explicit instead of accepting arbitrary user-supplied
# function names.
_ABSTRACT_FUNCTION_NAMES: frozenset[str] = frozenset("abcdfghjklmnopqrstuvwxyz")


def unsupported_function_name(
    expression: Expression,
    *,
    allow_abstract: bool = False,
    allowed_names: frozenset[str] = frozenset(),
) -> str | None:
    """Return the first function name outside the explicit supported registry."""
    if isinstance(expression, FunctionCall):
        name = str(expression.name)
        abstract = allow_abstract and name in _ABSTRACT_FUNCTION_NAMES
        if name not in _SUPPORTED_FUNCTION_NAMES and name not in allowed_names and not abstract:
            return name
        for argument in expression.arguments:
            unsupported = unsupported_function_name(
                argument,
                allow_abstract=allow_abstract,
                allowed_names=allowed_names,
            )
            if unsupported is not None:
                return unsupported
        return None
    if isinstance(expression, UnaryExpression):
        return unsupported_function_name(
            expression.operand,
            allow_abstract=allow_abstract,
            allowed_names=allowed_names,
        )
    if isinstance(expression, BinaryExpression):
        return unsupported_function_name(
            expression.left,
            allow_abstract=allow_abstract,
            allowed_names=allowed_names,
        ) or unsupported_function_name(
            expression.right,
            allow_abstract=allow_abstract,
            allowed_names=allowed_names,
        )
    if isinstance(expression, Relation):
        return unsupported_function_name(
            expression.left,
            allow_abstract=allow_abstract,
            allowed_names=allowed_names,
        ) or unsupported_function_name(
            expression.right,
            allow_abstract=allow_abstract,
            allowed_names=allowed_names,
        )
    if isinstance(expression, SequenceExpression):
        for item in expression.items:
            unsupported = unsupported_function_name(
                item,
                allow_abstract=allow_abstract,
                allowed_names=allowed_names,
            )
            if unsupported is not None:
                return unsupported
    return None


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
                return sp.Mul(left, sp.Pow(right, -1, evaluate=False), evaluate=False)
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
        name = str(expression.name)
        if name in {Operation.DIFFERENTIATE.value, "Derivative"}:
            return self._derivative_to_sympy(expression, name)
        arguments = tuple(self.to_sympy(argument) for argument in expression.arguments)
        unary = _UNARY_FUNCTIONS.get(name)
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
        if name == "root":
            if len(arguments) != 2:
                raise QueryError("root expects a radicand and an index")
            reciprocal_index = sp.Pow(arguments[1], -1, evaluate=False)
            return sp.Pow(arguments[0], reciprocal_index, evaluate=False)
        if name == "matrix":
            raise QueryError("matrix is only valid as an argument to a matrix operation")
        return sp.Function(name)(*arguments)

    def _derivative_to_sympy(self, expression: FunctionCall, name: str) -> sp.Basic:
        """Convert ``diff(...)`` or SymPy-style ``Derivative(...)`` syntax."""
        if len(expression.arguments) not in {2, 3}:
            raise QueryError(f"nested {name} expects two or three arguments")
        target = self.to_sympy(expression.arguments[0])
        variable_argument = expression.arguments[1]
        if isinstance(variable_argument, SequenceExpression):
            if len(expression.arguments) != 2:
                raise QueryError(f"{name} cannot combine a tuple with a separate order")
            if len(variable_argument.items) != 2:
                raise QueryError(f"{name} with a tuple expects a variable and an order")
            variable = self.to_sympy(
                expect_symbol(variable_argument.items[0], role="derivative variable")
            )
            order = expect_integer(variable_argument.items[1], role="derivative order")
            return sp.diff(target, variable, order)
        variable = self.to_sympy(variable_argument)
        order = 1
        if len(expression.arguments) == 3:
            order = expect_integer(expression.arguments[2], role="derivative order")
        return sp.diff(target, variable, order)

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

    @staticmethod
    def _is_simple_multiplicative_factor(value: sp.Basic) -> bool:
        """Reject factors whose textual form introduces a division or rational wrapper."""
        if isinstance(value, sp.Rational):
            return value.q == 1
        if isinstance(value, sp.Float):
            return False
        if value.func is sp.Pow:
            exponent = value.args[1]
            return isinstance(exponent, sp.Integer) and exponent > 0
        if value.func is sp.Mul:
            return all(
                SympyConverter._is_simple_multiplicative_factor(factor)
                for factor in value.as_ordered_factors()
                if factor != sp.Integer(1)
            )
        return True

    def _simple_from_sympy(self, value: sp.Basic) -> Expression | None:
        """Convert a simple algebraic/trigonometric node without reparsing text."""
        if isinstance(value, sp.Integer):
            return Number(value=Fraction(int(value.p), int(value.q)))
        if isinstance(value, sp.Rational):
            return Number(value=Fraction(int(value.p), int(value.q)))
        if isinstance(value, sp.Float):
            return ApproximateNumber(text=str(value))
        if isinstance(value, sp.Symbol):
            return Symbol(name=Identifier(str(value)))
        if value.func is sp.Add:
            if value.has(sp.Symbol("C")):
                return None
            terms = value.as_ordered_terms()
            if (
                not terms
                or any(term.could_extract_minus_sign() for term in terms)
                or any(isinstance(term, sp.Rational) and term.q != 1 for term in terms)
            ):
                return None
            expression = self.from_sympy(terms[0])
            for term in terms[1:]:
                expression = BinaryExpression(
                    operator=BinaryOperator.ADD,
                    left=expression,
                    right=self.from_sympy(term),
                )
            return expression
        if value.func is sp.Mul:
            coefficient, _rest = value.as_coeff_Mul()
            if not isinstance(coefficient, sp.Integer) or coefficient.is_negative is True:
                return None
            factors = tuple(
                factor for factor in value.as_ordered_factors() if factor != sp.Integer(1)
            )
            if not factors or any(
                not self._is_simple_multiplicative_factor(factor) for factor in factors
            ):
                return None
            expression = self.from_sympy(factors[0])
            for factor in factors[1:]:
                expression = BinaryExpression(
                    operator=BinaryOperator.MULTIPLY,
                    left=expression,
                    right=self.from_sympy(factor),
                )
            return expression
        if value.func is sp.Pow:
            base, exponent = value.args
            if not isinstance(exponent, sp.Integer) or exponent <= 0:
                return None
            return BinaryExpression(
                operator=BinaryOperator.POWER,
                left=self.from_sympy(base),
                right=Number(value=Fraction(int(exponent.p), int(exponent.q))),
            )
        functions: dict[object, str] = {
            sp.sin: "sin",
            sp.cos: "cos",
            sp.tan: "tan",
            sp.asin: "asin",
            sp.acos: "acos",
            sp.atan: "atan",
            sp.sinh: "sinh",
            sp.cosh: "cosh",
            sp.tanh: "tanh",
            sp.asinh: "asinh",
            sp.exp: "exp",
            sp.gamma: "gamma",
        }
        function_name = functions.get(value.func)
        if function_name is not None and len(value.args) == 1:
            return FunctionCall(
                name=Identifier(function_name),
                arguments=(self.from_sympy(value.args[0]),),
            )
        return None

    def from_sympy(self, value: sp.Basic) -> Expression:
        """Convert a SymPy scalar to the closest public AST expression."""
        integration_constant = sp.Symbol("C")
        if value.func == sp.factorial and len(value.args) == 1:
            return UnaryExpression(
                operator=UnaryOperator.FACTORIAL,
                operand=self.from_sympy(value.args[0]),
            )
        if value.func == sp.Abs and len(value.args) == 1:
            return FunctionCall(
                name=Identifier("abs"),
                arguments=(self.from_sympy(value.args[0]),),
            )
        if value.func == sp.log and len(value.args) == 1:
            return FunctionCall(
                name=Identifier("log"),
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
        simple = self._simple_from_sympy(value)
        if simple is not None:
            return simple
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
