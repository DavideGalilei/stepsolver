"""Public solver orchestration."""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import isfinite

from stepsolver.ast import (
    ApproximateNumber,
    Expression,
    FunctionCall,
    Number,
    Operation,
    Query,
    SolutionDomain,
    Symbol,
)
from stepsolver.backend import SymbolicBackend
from stepsolver.errors import QueryError
from stepsolver.parser import parse, parse_expression
from stepsolver.results import SolveResult
from stepsolver.sympy_backend import SympyBackend


@dataclass(frozen=True, slots=True, kw_only=True)
class Solver:
    """Parse and solve StepSolver queries through a symbolic backend."""

    backend: SymbolicBackend = field(default_factory=SympyBackend)

    def solve(
        self,
        problem: str | Query,
        *,
        domain: SolutionDomain | str | None = None,
    ) -> SolveResult:
        """Solve source text or an already parsed query.

        ``domain`` optionally restricts an exact solve to real or complex values.
        Omitting it preserves the documented per-operation default.
        """
        query = parse(problem) if isinstance(problem, str) else problem
        requested = self._coerce_domain(domain) if domain is not None else None
        if requested is not None:
            if query.domain is not None and query.domain is not requested:
                raise QueryError("domain conflicts with the domain already recorded on the query")
            if query.domain is None:
                query = replace(query, domain=requested)
        return self.backend.solve(query)

    @staticmethod
    def _coerce_domain(domain: SolutionDomain | str) -> SolutionDomain:
        """Validate a caller-supplied ambient domain."""
        if isinstance(domain, SolutionDomain):
            return domain
        try:
            return SolutionDomain(domain.strip().lower())
        except ValueError as error:
            raise QueryError("domain must be 'real' or 'complex'") from error

    def nsolve(
        self,
        problem: str | Query,
        *,
        var: str | Symbol | None = None,
        guess: float | int | Fraction | str | None = None,
    ) -> SolveResult:
        """Find real roots numerically with a worked Newton derivation.

        ``problem`` may be an equation such as ``"cos(x) = x"`` or a residual
        expression such as ``"cos(x) - x"``.  A bare equation or residual needs
        an explicit ``var``; an ``nsolve(...)`` query supplies its own variable
        and optional guess.

        The initial guess may be an integer, float, exact fraction, or a short
        real numeric string.  With no guess, polynomial residuals use real-root
        isolation; a general transcendental residual is rejected with a reason
        instead of pretending that its root is unique or automatic.  The
        implementation solves over the real domain only, rejects non-real
        candidates, verifies each returned root by substitution, uses a
        residual tolerance of ``1e-12``, and bounds Newton's method to 50
        iterations.  Results display 15 significant digits.
        """
        return self.backend.solve(self._numeric_query(problem, var=var, guess=guess))

    def _numeric_query(
        self,
        problem: str | Query,
        *,
        var: str | Symbol | None,
        guess: float | int | Fraction | str | None,
    ) -> Query:
        """Build a first-class ``nsolve`` query from the convenience API."""
        if isinstance(problem, Query):
            if problem.operation is not Operation.NSOLVE:
                raise QueryError("nsolve expects a numeric query")
            if var is not None or guess is not None:
                raise QueryError("var and guess cannot accompany an nsolve query")
            return problem
        expression = parse_expression(problem)
        if isinstance(expression, FunctionCall) and str(expression.name) == Operation.NSOLVE.value:
            if var is not None or guess is not None:
                raise QueryError("var and guess cannot accompany an nsolve(...) input")
            return Query(
                operation=Operation.NSOLVE,
                arguments=expression.arguments,
                source=problem,
            )
        variable = self._numeric_variable(var)
        arguments: tuple[Expression, ...] = (expression, variable)
        if guess is not None:
            arguments = (*arguments, self._numeric_guess(guess))
        return Query(operation=Operation.NSOLVE, arguments=arguments, source=problem)

    @staticmethod
    def _numeric_variable(var: str | Symbol | None) -> Symbol:
        if var is None:
            raise QueryError("nsolve requires var='x' for a bare equation or residual")
        if isinstance(var, Symbol):
            return var
        candidate = parse_expression(var)
        if not isinstance(candidate, Symbol):
            raise QueryError("nsolve variable must be a single symbol")
        return candidate

    @staticmethod
    def _numeric_guess(guess: float | int | Fraction | str) -> Expression:
        if isinstance(guess, bool):
            raise QueryError("initial guess must be a real number, not a boolean")
        if isinstance(guess, int):
            return Number(value=Fraction(guess))
        if isinstance(guess, Fraction):
            return Number(value=guess)
        if isinstance(guess, float):
            if not isfinite(guess):
                raise QueryError("initial guess must be finite")
            return ApproximateNumber(text=repr(guess))
        return parse_expression(guess)
