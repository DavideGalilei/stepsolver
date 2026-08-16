"""Strictly typed step-by-step symbolic mathematics."""

from stepsolver.ast import (
    Expression,
    Operation,
    Query,
)
from stepsolver.backend import SymbolicBackend
from stepsolver.errors import BackendError, ParseError, QueryError, StepSolverError
from stepsolver.formatter import format_ascii, format_expression
from stepsolver.latex import format_latex_expression, format_latex_value
from stepsolver.parser import parse, parse_expression
from stepsolver.presentation import SolvePayload, StepNotePayload, StepPayload, solve_payload
from stepsolver.results import (
    BooleanValue,
    DivergenceKind,
    DivergentResult,
    ExactResult,
    MappingEntry,
    MappingValue,
    MathValue,
    MatrixValue,
    ScalarValue,
    SequenceValue,
    SolutionStep,
    SolveResult,
    StepNote,
    UnsolvedResult,
    Verification,
    VerificationMethod,
)
from stepsolver.solver import Solver
from stepsolver.sympy_backend import SympyBackend

__all__ = [
    "BackendError",
    "BooleanValue",
    "DivergenceKind",
    "DivergentResult",
    "ExactResult",
    "Expression",
    "MappingEntry",
    "MappingValue",
    "MathValue",
    "MatrixValue",
    "Operation",
    "ParseError",
    "Query",
    "QueryError",
    "ScalarValue",
    "SequenceValue",
    "SolutionStep",
    "SolvePayload",
    "SolveResult",
    "Solver",
    "StepNote",
    "StepNotePayload",
    "StepPayload",
    "StepSolverError",
    "SymbolicBackend",
    "SympyBackend",
    "UnsolvedResult",
    "Verification",
    "VerificationMethod",
    "format_ascii",
    "format_expression",
    "format_latex_expression",
    "format_latex_value",
    "parse",
    "parse_expression",
    "solve_payload",
]
