"""Backend contracts used by the solver orchestration layer."""

from typing import Protocol

from stepsolver.ast import Query
from stepsolver.results import SolveResult


class SymbolicBackend(Protocol):
    """A symbolic engine capable of solving a validated query."""

    def solve(self, query: Query) -> SolveResult:
        """Solve a query without leaking backend-native objects."""
        ...
