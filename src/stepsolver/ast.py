"""Immutable mathematical abstract syntax tree nodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from fractions import Fraction

Identifier = NewType("Identifier", str)


class ConstantName(Enum):
    """Built-in mathematical constants."""

    PI = "pi"
    E = "e"
    IMAGINARY = "i"
    INFINITY = "oo"


class UnaryOperator(Enum):
    """Unary expression operators."""

    POSITIVE = "+"
    NEGATIVE = "-"
    FACTORIAL = "!"


class BinaryOperator(Enum):
    """Binary arithmetic operators."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "^"


class RelationOperator(Enum):
    """Binary relation operators."""

    EQUAL = "="
    NOT_EQUAL = "!="
    LESS = "<"
    LESS_EQUAL = "<="
    GREATER = ">"
    GREATER_EQUAL = ">="


class Operation(Enum):
    """Supported top-level solver operations."""

    SIMPLIFY = "simplify"
    EXPAND = "expand"
    FACTOR = "factor"
    CANCEL = "cancel"
    APART = "apart"
    SOLVE = "solve"
    SOLVE_INEQUALITY = "solve_inequality"
    DIFFERENTIATE = "diff"
    INTEGRATE = "integrate"
    CONTOUR_INTEGRATE = "contour_integrate"
    LIMIT = "limit"
    SERIES = "series"
    SUM = "sum"
    PRODUCT = "product"
    MATRIX = "matrix"
    DETERMINANT = "det"
    INVERSE = "inverse"
    RANK = "rank"
    RREF = "rref"
    EIGENVALUES = "eigenvalues"
    DSOLVE = "dsolve"
    RSOLVE = "rsolve"
    LAPLACE = "laplace"
    INVERSE_LAPLACE = "inverse_laplace"
    FOURIER = "fourier"
    INVERSE_FOURIER = "inverse_fourier"
    GCD = "gcd"
    LCM = "lcm"
    IS_PRIME = "is_prime"
    PRIME_FACTORS = "prime_factors"
    BINOMIAL = "binomial"
    PERMUTATIONS = "permutations"
    COMBINATIONS = "combinations"
    NUMERIC = "numeric"


@dataclass(frozen=True, slots=True, kw_only=True)
class Number:
    """An exact rational number."""

    value: Fraction


@dataclass(frozen=True, slots=True, kw_only=True)
class ApproximateNumber:
    """A decimal approximation with explicit displayed precision."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Symbol:
    """A named mathematical symbol."""

    name: Identifier


@dataclass(frozen=True, slots=True, kw_only=True)
class Constant:
    """A built-in mathematical constant."""

    name: ConstantName


@dataclass(frozen=True, slots=True, kw_only=True)
class UnaryExpression:
    """An expression with one operator and operand."""

    operator: UnaryOperator
    operand: Expression


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryExpression:
    """An expression with two operands."""

    operator: BinaryOperator
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionCall:
    """A mathematical function invocation."""

    name: Identifier
    arguments: tuple[Expression, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Relation:
    """A relation between two expressions."""

    operator: RelationOperator
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceExpression:
    """An ordered expression sequence used for systems and matrices."""

    items: tuple[Expression, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class OpaqueExpression:
    """A canonical backend expression not representable by structural nodes yet."""

    text: str


type Expression = (
    Number
    | ApproximateNumber
    | Symbol
    | Constant
    | UnaryExpression
    | BinaryExpression
    | FunctionCall
    | Relation
    | SequenceExpression
    | OpaqueExpression
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    """A validated top-level solver request."""

    operation: Operation
    arguments: tuple[Expression, ...]
    source: str
