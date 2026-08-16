"""Public solver orchestration."""

from dataclasses import dataclass, field

from stepsolver.ast import Query
from stepsolver.backend import SymbolicBackend
from stepsolver.parser import parse
from stepsolver.results import SolveResult
from stepsolver.sympy_backend import SympyBackend


@dataclass(frozen=True, slots=True, kw_only=True)
class Solver:
    """Parse and solve StepSolver queries through a symbolic backend."""

    backend: SymbolicBackend = field(default_factory=SympyBackend)

    def solve(self, problem: str | Query) -> SolveResult:
        """Solve source text or an already parsed query."""
        query = parse(problem) if isinstance(problem, str) else problem
        return self.backend.solve(query)
