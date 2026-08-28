"""Pure limits and typed payloads for isolated StepSolver workers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Constant,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    OpaqueExpression,
    Operation,
    Query,
    Relation,
    SequenceExpression,
    Symbol,
    UnaryExpression,
)
from stepsolver.errors import LimitError, LimitViolation, ParseError
from stepsolver.formatter import format_expression, format_value
from stepsolver.parser import parse
from stepsolver.presentation import (
    StepConstraintPayload,
    StepNotePayload,
    StepPayload,
    solve_payload,
)
from stepsolver.results import ExactResult, SolveResult
from stepsolver.solver import Solver


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedLimits:
    """Independent library ceilings for one isolated solve."""

    request_bytes: int = 4_096
    expression_characters: int = 2_048
    ast_nodes: int = 256
    ast_depth: int = 32
    symbols: int = 64
    function_name_characters: int = 64
    integer_digits: int = 100
    absolute_numeric_exponent: int = 100
    function_arguments: int = 32
    sequence_items: int = 64
    matrix_rows: int = 8
    matrix_columns: int = 8
    finite_range_terms: int = 10_000
    steps: int = 64
    step_notes: int = 16
    step_constraints: int = 16
    rendered_field_characters: int = 8_192
    response_bytes: int = 65_536


DEFAULT_ISOLATED_LIMITS = IsolatedLimits()


class _SolveRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str

    @field_validator("expression")
    @classmethod
    def require_ascii_expression(cls, value: str) -> str:
        if not value.isascii():
            message = "expression must contain only ASCII characters"
            raise ValueError(message)
        return value


class IsolatedErrorCode(Enum):
    """Stable failures exposed by the one-request worker protocol."""

    INVALID_REQUEST = "invalid_request"
    INPUT_LIMIT = "input_limit"
    EXPRESSION_LIMIT = "expression_limit"
    RESULT_LIMIT = "result_limit"
    SOLVER_FAILED = "solver_failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedSolveRequest:
    """A parsed request whose complete AST satisfies isolated-runner limits."""

    expression: str
    query: Query


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedNote:
    """One bounded identity or substitution supporting a step."""

    label: str
    expression: str
    latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedConstraint:
    """One bounded domain restriction introduced by a step."""

    explanation: str
    expression: str
    latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedStep:
    """One bounded, provider-neutral worked step."""

    rule: str
    explanation: str
    before: str
    after: str
    before_latex: str
    after_latex: str
    notes: tuple[IsolatedNote, ...]
    constraints: tuple[IsolatedConstraint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedSolution:
    """Bounded solver output returned to an agent."""

    normalized_expression: str
    status: Literal["exact", "divergent", "undefined", "unsolved"]
    result: str | None
    latex: str | None
    reason: str | None
    steps: tuple[IsolatedStep, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedSuccess:
    """Successful runner response."""

    ok: Literal[True]
    solution: IsolatedSolution

    def as_dict(self) -> dict[str, object]:
        """Return standard containers for compact JSON serialization."""
        return asdict(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class IsolatedFailure:
    """Stable runner failure without expression or traceback data."""

    ok: Literal[False]
    code: IsolatedErrorCode
    message: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible failure object."""
        return {"ok": self.ok, "error": {"code": self.code.value, "message": self.message}}


def _children(expression: Expression) -> tuple[Expression, ...]:
    match expression:
        case UnaryExpression(operand=operand):
            return (operand,)
        case BinaryExpression(left=left, right=right) | Relation(left=left, right=right):
            return (left, right)
        case FunctionCall(arguments=arguments):
            return arguments
        case SequenceExpression(items=items):
            return items
        case Number() | ApproximateNumber() | Symbol() | Constant() | OpaqueExpression():
            return ()


def _decimal_digits(value: int) -> int:
    return len(str(abs(value)))


def _validate_node(expression: Expression, limits: IsolatedLimits) -> str | None:
    if isinstance(expression, Number) and (
        max(
            _decimal_digits(expression.value.numerator),
            _decimal_digits(expression.value.denominator),
        )
        > limits.integer_digits
    ):
        raise LimitError(LimitViolation.INTEGER_DIGITS)
    if isinstance(expression, Symbol):
        return str(expression.name)
    if isinstance(expression, FunctionCall):
        if len(expression.arguments) > limits.function_arguments:
            raise LimitError(LimitViolation.FUNCTION_ARGUMENTS)
        if len(expression.name) > limits.function_name_characters:
            raise LimitError(LimitViolation.FUNCTION_NAME)
    if isinstance(expression, SequenceExpression) and len(expression.items) > limits.sequence_items:
        raise LimitError(LimitViolation.SEQUENCE_ITEMS)
    if (
        isinstance(expression, BinaryExpression)
        and expression.operator is BinaryOperator.POWER
        and isinstance(expression.right, Number)
        and abs(expression.right.value) > limits.absolute_numeric_exponent
    ):
        raise LimitError(LimitViolation.NUMERIC_EXPONENT)
    return None


def _validate_ast(query: Query, limits: IsolatedLimits) -> None:
    stack = [(argument, 1) for argument in reversed(query.arguments)]
    node_count = 0
    symbols: set[str] = set()
    while stack:
        expression, depth = stack.pop()
        node_count += 1
        if node_count > limits.ast_nodes:
            raise LimitError(LimitViolation.AST_NODES)
        if depth > limits.ast_depth:
            raise LimitError(LimitViolation.AST_DEPTH)
        symbol = _validate_node(expression, limits)
        if symbol is not None:
            symbols.add(symbol)
            if len(symbols) > limits.symbols:
                raise LimitError(LimitViolation.SYMBOLS)
        stack.extend((child, depth + 1) for child in reversed(_children(expression)))


def _integer_value(expression: Expression) -> int | None:
    if isinstance(expression, Number) and expression.value.denominator == 1:
        return expression.value.numerator
    return None


def _validate_finite_range(query: Query, limits: IsolatedLimits) -> None:
    expected_arguments = 4
    if (
        query.operation not in {Operation.SUM, Operation.PRODUCT}
        or len(query.arguments) != expected_arguments
    ):
        return
    lower = _integer_value(query.arguments[2])
    upper = _integer_value(query.arguments[3])
    if lower is None or upper is None:
        return
    terms = max(0, upper - lower + 1)
    if terms > limits.finite_range_terms:
        raise LimitError(LimitViolation.FINITE_RANGE)


def _matrix_rows(expression: Expression) -> tuple[SequenceExpression, ...] | None:
    if not isinstance(expression, SequenceExpression):
        return None
    rows = tuple(item for item in expression.items if isinstance(item, SequenceExpression))
    return rows if len(rows) == len(expression.items) else None


def _validate_matrix(query: Query, limits: IsolatedLimits) -> None:
    matrix_operations = {
        Operation.MATRIX,
        Operation.DETERMINANT,
        Operation.INVERSE,
        Operation.RANK,
        Operation.RREF,
        Operation.EIGENVALUES,
    }
    if query.operation not in matrix_operations or len(query.arguments) != 1:
        return
    rows = _matrix_rows(query.arguments[0])
    if rows is None:
        return
    columns = len(rows[0].items) if rows else 0
    if len(rows) > limits.matrix_rows or columns > limits.matrix_columns:
        raise LimitError(LimitViolation.MATRIX_DIMENSIONS)
    if any(len(row.items) != columns for row in rows):
        raise LimitError(LimitViolation.MATRIX_SHAPE)


def validate_solve_request(
    expression: str,
    limits: IsolatedLimits = DEFAULT_ISOLATED_LIMITS,
) -> ValidatedSolveRequest:
    """Parse one ASCII expression and prove it satisfies every input limit."""
    if len(expression) > limits.expression_characters:
        raise LimitError(LimitViolation.EXPRESSION_CHARACTERS)
    query = parse(expression)
    _validate_ast(query, limits)
    _validate_finite_range(query, limits)
    _validate_matrix(query, limits)
    return ValidatedSolveRequest(expression=expression, query=query)


def parse_isolated_request(
    document: bytes,
    limits: IsolatedLimits = DEFAULT_ISOLATED_LIMITS,
) -> ValidatedSolveRequest:
    """Parse the exact JSON request shape and validate its mathematical payload."""
    if len(document) > limits.request_bytes:
        raise LimitError(LimitViolation.REQUEST_BYTES)
    try:
        request = _SolveRequestPayload.model_validate_json(document)
    except ValidationError as error:
        message = "request must contain one ASCII expression"
        raise ParseError(message, position=0) from error
    return validate_solve_request(request.expression, limits)


def _normalized_expression(query: Query) -> str:
    if query.operation is Operation.SIMPLIFY:
        return format_expression(query.arguments[0])
    call = FunctionCall(name=Identifier(query.operation.value), arguments=query.arguments)
    return format_expression(call)


def _bounded_text(value: str, limits: IsolatedLimits) -> str:
    if len(value) > limits.rendered_field_characters:
        raise LimitError(LimitViolation.RENDERED_FIELD)
    return value


def _bounded_optional_text(value: str | None, limits: IsolatedLimits) -> str | None:
    return None if value is None else _bounded_text(value, limits)


def _isolated_note(note: StepNotePayload, limits: IsolatedLimits) -> IsolatedNote:
    return IsolatedNote(
        label=_bounded_text(note.label, limits),
        expression=_bounded_text(note.expression_ascii, limits),
        latex=_bounded_text(note.expression_latex, limits),
    )


def _isolated_constraint(
    constraint: StepConstraintPayload,
    limits: IsolatedLimits,
) -> IsolatedConstraint:
    return IsolatedConstraint(
        explanation=_bounded_text(constraint.explanation, limits),
        expression=_bounded_text(constraint.expression_ascii, limits),
        latex=_bounded_text(constraint.expression_latex, limits),
    )


def _isolated_step(step: StepPayload, limits: IsolatedLimits) -> IsolatedStep:
    return IsolatedStep(
        rule=_bounded_text(step.rule, limits),
        explanation=_bounded_text(step.explanation, limits),
        before=_bounded_text(step.before_ascii, limits),
        after=_bounded_text(step.after_ascii, limits),
        before_latex=_bounded_text(step.before_latex, limits),
        after_latex=_bounded_text(step.after_latex, limits),
        notes=tuple(_isolated_note(note, limits) for note in step.notes),
        constraints=tuple(
            _isolated_constraint(constraint, limits) for constraint in step.introduced_constraints
        ),
    )


def bounded_solution(
    request: ValidatedSolveRequest,
    result: SolveResult,
    limits: IsolatedLimits = DEFAULT_ISOLATED_LIMITS,
) -> IsolatedSolution:
    """Calculate a bounded agent-facing response without performing I/O."""
    payload = solve_payload(result)
    if len(payload.steps) > limits.steps:
        raise LimitError(LimitViolation.STEPS)
    if any(len(step.notes) > limits.step_notes for step in payload.steps):
        raise LimitError(LimitViolation.STEP_NOTES)
    if any(len(step.introduced_constraints) > limits.step_constraints for step in payload.steps):
        raise LimitError(LimitViolation.STEP_CONSTRAINTS)
    solution = IsolatedSolution(
        normalized_expression=_bounded_text(_normalized_expression(request.query), limits),
        status=payload.status,
        result=(
            _bounded_text(format_value(result.value), limits)
            if isinstance(result, ExactResult)
            else None
        ),
        latex=_bounded_optional_text(payload.result_latex, limits),
        reason=_bounded_optional_text(payload.reason, limits),
        steps=tuple(_isolated_step(step, limits) for step in payload.steps),
    )
    encoded = json.dumps(
        IsolatedSuccess(ok=True, solution=solution).as_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > limits.response_bytes:
        raise LimitError(LimitViolation.RESPONSE_BYTES)
    return solution


def solve_isolated(
    request: ValidatedSolveRequest,
    *,
    solver: Solver | None = None,
    limits: IsolatedLimits = DEFAULT_ISOLATED_LIMITS,
) -> IsolatedSolution:
    """Execute one previously validated request and bound its result."""
    result = (solver or Solver()).solve(request.query)
    return bounded_solution(request, result, limits)
