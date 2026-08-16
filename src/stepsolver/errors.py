"""StepSolver exception hierarchy."""


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
