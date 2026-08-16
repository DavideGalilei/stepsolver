"""Safe ASCII lexer and parser for StepSolver queries."""

from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction

from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Constant,
    ConstantName,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.errors import ParseError


class _TokenKind(Enum):
    NUMBER = auto()
    IDENTIFIER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()
    FACTORIAL = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    END = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Token:
    kind: _TokenKind
    text: str
    position: int


_SINGLE_TOKENS: dict[str, _TokenKind] = {
    "+": _TokenKind.PLUS,
    "-": _TokenKind.MINUS,
    "*": _TokenKind.STAR,
    "/": _TokenKind.SLASH,
    "^": _TokenKind.POWER,
    "!": _TokenKind.FACTORIAL,
    "=": _TokenKind.EQUAL,
    "<": _TokenKind.LESS,
    ">": _TokenKind.GREATER,
    "(": _TokenKind.LEFT_PAREN,
    ")": _TokenKind.RIGHT_PAREN,
    "[": _TokenKind.LEFT_BRACKET,
    "]": _TokenKind.RIGHT_BRACKET,
    ",": _TokenKind.COMMA,
}

_DOUBLE_TOKENS: dict[str, _TokenKind] = {
    "**": _TokenKind.POWER,
    "!=": _TokenKind.NOT_EQUAL,
    "<=": _TokenKind.LESS_EQUAL,
    ">=": _TokenKind.GREATER_EQUAL,
}

_CONSTANTS: dict[str, ConstantName] = {constant.value: constant for constant in ConstantName}
_OPERATIONS: dict[str, Operation] = {operation.value: operation for operation in Operation}


def _tokenize(source: str) -> tuple[_Token, ...]:
    if not source.isascii():
        raise ParseError("input must contain ASCII characters only", position=0)
    tokens: list[_Token] = []
    position = 0
    while position < len(source):
        character = source[position]
        if character.isspace():
            position += 1
            continue
        pair = source[position : position + 2]
        pair_kind = _DOUBLE_TOKENS.get(pair)
        if pair_kind is not None:
            tokens.append(_Token(kind=pair_kind, text=pair, position=position))
            position += 2
            continue
        if character.isdigit() or (
            character == "." and position + 1 < len(source) and source[position + 1].isdigit()
        ):
            start = position
            dots = 0
            while position < len(source):
                current = source[position]
                if current == ".":
                    dots += 1
                    if dots > 1:
                        break
                elif not current.isdigit():
                    break
                position += 1
            text = source[start:position]
            if text == ".":
                raise ParseError("expected digits around decimal point", position=start)
            tokens.append(_Token(kind=_TokenKind.NUMBER, text=text, position=start))
            continue
        if character.isalpha() or character == "_":
            start = position
            position += 1
            while position < len(source):
                current = source[position]
                if not (current.isalnum() or current == "_"):
                    break
                position += 1
            tokens.append(
                _Token(
                    kind=_TokenKind.IDENTIFIER,
                    text=source[start:position],
                    position=start,
                )
            )
            continue
        single_kind = _SINGLE_TOKENS.get(character)
        if single_kind is None:
            raise ParseError(f"unexpected character {character!r}", position=position)
        tokens.append(_Token(kind=single_kind, text=character, position=position))
        position += 1
    tokens.append(_Token(kind=_TokenKind.END, text="", position=len(source)))
    return tuple(tokens)


class _Parser:
    def __init__(self, source: str, *, allow_implicit_multiplication: bool) -> None:
        self._source = source
        self._tokens = _tokenize(source)
        self._index = 0
        self._allow_implicit_multiplication = allow_implicit_multiplication

    @property
    def _current(self) -> _Token:
        return self._tokens[self._index]

    def parse(self) -> Expression:
        expression = self._parse_relation()
        if self._current.kind is not _TokenKind.END:
            raise ParseError("unexpected token", position=self._current.position)
        return expression

    def _advance(self) -> _Token:
        token = self._current
        self._index += 1
        return token

    def _accept(self, kind: _TokenKind) -> _Token | None:
        if self._current.kind is kind:
            return self._advance()
        return None

    def _expect(self, kind: _TokenKind, message: str) -> _Token:
        token = self._accept(kind)
        if token is None:
            raise ParseError(message, position=self._current.position)
        return token

    def _parse_relation(self) -> Expression:
        left = self._parse_additive()
        relation_operators: dict[_TokenKind, RelationOperator] = {
            _TokenKind.EQUAL: RelationOperator.EQUAL,
            _TokenKind.NOT_EQUAL: RelationOperator.NOT_EQUAL,
            _TokenKind.LESS: RelationOperator.LESS,
            _TokenKind.LESS_EQUAL: RelationOperator.LESS_EQUAL,
            _TokenKind.GREATER: RelationOperator.GREATER,
            _TokenKind.GREATER_EQUAL: RelationOperator.GREATER_EQUAL,
        }
        operator = relation_operators.get(self._current.kind)
        if operator is None:
            return left
        self._advance()
        right = self._parse_additive()
        return Relation(operator=operator, left=left, right=right)

    def _parse_additive(self) -> Expression:
        expression = self._parse_multiplicative()
        while self._current.kind in {_TokenKind.PLUS, _TokenKind.MINUS}:
            kind = self._advance().kind
            right = self._parse_multiplicative()
            operator = BinaryOperator.ADD if kind is _TokenKind.PLUS else BinaryOperator.SUBTRACT
            expression = BinaryExpression(operator=operator, left=expression, right=right)
        return expression

    def _parse_multiplicative(self) -> Expression:
        expression = self._parse_unary()
        implicit_starts = {
            _TokenKind.IDENTIFIER,
            _TokenKind.LEFT_BRACKET,
            _TokenKind.LEFT_PAREN,
        }
        while True:
            kind = self._current.kind
            if kind in {_TokenKind.STAR, _TokenKind.SLASH}:
                self._advance()
                operator = (
                    BinaryOperator.MULTIPLY if kind is _TokenKind.STAR else BinaryOperator.DIVIDE
                )
            elif self._allow_implicit_multiplication and kind in implicit_starts:
                operator = BinaryOperator.MULTIPLY
            else:
                break
            right = self._parse_unary()
            expression = BinaryExpression(operator=operator, left=expression, right=right)
        return expression

    def _parse_unary(self) -> Expression:
        if self._accept(_TokenKind.PLUS) is not None:
            operand = self._parse_unary()
            if isinstance(operand, Number):
                return operand
            return UnaryExpression(operator=UnaryOperator.POSITIVE, operand=operand)
        if self._accept(_TokenKind.MINUS) is not None:
            operand = self._parse_unary()
            if isinstance(operand, Number):
                return Number(value=-operand.value)
            return UnaryExpression(operator=UnaryOperator.NEGATIVE, operand=operand)
        return self._parse_power()

    def _parse_power(self) -> Expression:
        expression = self._parse_postfix()
        if self._accept(_TokenKind.POWER) is not None:
            expression = BinaryExpression(
                operator=BinaryOperator.POWER,
                left=expression,
                right=self._parse_unary(),
            )
        return expression

    def _parse_postfix(self) -> Expression:
        expression = self._parse_primary()
        while self._accept(_TokenKind.FACTORIAL) is not None:
            expression = UnaryExpression(operator=UnaryOperator.FACTORIAL, operand=expression)
        return expression

    def _parse_primary(self) -> Expression:
        token = self._current
        if token.kind is _TokenKind.NUMBER:
            self._advance()
            return Number(value=Fraction(token.text))
        if token.kind is _TokenKind.IDENTIFIER:
            self._advance()
            if self._accept(_TokenKind.LEFT_PAREN) is not None:
                return self._parse_call(token)
            constant = _CONSTANTS.get(token.text)
            if constant is not None:
                return Constant(name=constant)
            return Symbol(name=Identifier(token.text))
        if self._accept(_TokenKind.LEFT_PAREN) is not None:
            expression = self._parse_relation()
            self._expect(_TokenKind.RIGHT_PAREN, "expected ')'")
            return expression
        if self._accept(_TokenKind.LEFT_BRACKET) is not None:
            return self._parse_sequence()
        raise ParseError("expected an expression", position=token.position)

    def _parse_call(self, name: _Token) -> FunctionCall:
        arguments: list[Expression] = []
        if self._accept(_TokenKind.RIGHT_PAREN) is None:
            while True:
                arguments.append(self._parse_relation())
                if self._accept(_TokenKind.COMMA) is None:
                    break
            self._expect(_TokenKind.RIGHT_PAREN, "expected ')' after function arguments")
        return FunctionCall(name=Identifier(name.text), arguments=tuple(arguments))

    def _parse_sequence(self) -> SequenceExpression:
        items: list[Expression] = []
        if self._accept(_TokenKind.RIGHT_BRACKET) is None:
            while True:
                items.append(self._parse_relation())
                if self._accept(_TokenKind.COMMA) is None:
                    break
            self._expect(_TokenKind.RIGHT_BRACKET, "expected ']' after sequence")
        return SequenceExpression(items=tuple(items))


def parse_expression(source: str, *, allow_implicit_multiplication: bool = False) -> Expression:
    """Parse one ASCII mathematical expression."""
    if not source.strip():
        raise ParseError("input cannot be empty", position=0)
    return _Parser(source, allow_implicit_multiplication=allow_implicit_multiplication).parse()


def parse(source: str) -> Query:
    """Parse an ASCII solver query, defaulting bare expressions to simplification."""
    expression = parse_expression(source)
    if isinstance(expression, FunctionCall):
        operation = _OPERATIONS.get(expression.name)
        if operation is not None:
            return Query(operation=operation, arguments=expression.arguments, source=source)
    return Query(operation=Operation.SIMPLIFY, arguments=(expression,), source=source)
