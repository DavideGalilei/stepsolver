"""StepSolver exception hierarchy."""

from enum import Enum


class StepSolverError(Exception):
    """Base class for public StepSolver errors."""


class ParseError(StepSolverError):
    """Raised when ASCII mathematical input is syntactically invalid."""

    def __init__(self, message: str, *, position: int) -> None:
        """Create a parse error carrying its zero-based source position."""
        self.position = position
        super().__init__(f"{message} at position {position}")


class QueryError(StepSolverError):
    """Raised when a parsed operation has invalid arguments."""


class BackendError(StepSolverError):
    """Raised when the symbolic backend violates its typed contract."""


class LimitCategory(Enum):
    """Protocol layer whose declared bound was exceeded."""

    INPUT = "input"
    EXPRESSION = "expression"
    RESULT = "result"


class LimitViolation(Enum):
    """Stable, typed reasons an isolated solve can exceed its bounds."""

    REQUEST_BYTES = ("The request exceeds the byte limit.", LimitCategory.INPUT)
    EXPRESSION_CHARACTERS = (
        "The expression exceeds the character limit.",
        LimitCategory.EXPRESSION,
    )
    INTEGER_DIGITS = ("An exact integer exceeds the digit limit.", LimitCategory.EXPRESSION)
    FUNCTION_ARGUMENTS = (
        "A function call has too many arguments.",
        LimitCategory.EXPRESSION,
    )
    FUNCTION_NAME = ("A function name is too long.", LimitCategory.EXPRESSION)
    SEQUENCE_ITEMS = ("A sequence has too many items.", LimitCategory.EXPRESSION)
    NUMERIC_EXPONENT = ("A numeric exponent exceeds the limit.", LimitCategory.EXPRESSION)
    AST_NODES = ("The expression has too many syntax nodes.", LimitCategory.EXPRESSION)
    AST_DEPTH = ("The expression is nested too deeply.", LimitCategory.EXPRESSION)
    SYMBOLS = ("The expression has too many symbols.", LimitCategory.EXPRESSION)
    FINITE_RANGE = (
        "A finite sum or product exceeds the term limit.",
        LimitCategory.EXPRESSION,
    )
    MATRIX_DIMENSIONS = ("A matrix exceeds the dimension limit.", LimitCategory.EXPRESSION)
    MATRIX_SHAPE = ("Matrix rows must have equal length.", LimitCategory.EXPRESSION)
    RENDERED_FIELD = ("A rendered result field exceeds the limit.", LimitCategory.RESULT)
    STEPS = ("The result has too many steps.", LimitCategory.RESULT)
    STEP_NOTES = ("A result step has too many notes.", LimitCategory.RESULT)
    STEP_CONSTRAINTS = ("A result step has too many constraints.", LimitCategory.RESULT)
    RESPONSE_BYTES = ("The serialized result exceeds the byte limit.", LimitCategory.RESULT)

    def __init__(self, message: str, category: LimitCategory) -> None:
        """Attach stable caller text and a machine-facing category."""
        self.message = message
        self.category = category


class LimitError(StepSolverError):
    """Raised when an isolated-runner input or result exceeds a declared limit."""

    def __init__(self, violation: LimitViolation) -> None:
        """Create an error from one typed limit violation."""
        self.violation = violation
        super().__init__(violation.message)
