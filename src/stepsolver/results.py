"""Typed solver values, verification records, steps, and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stepsolver.ast import Expression, Query


class VerificationMethod(Enum):
    """Machine checks used to validate a displayed step."""

    SYMBOLIC_EQUIVALENCE = "symbolic equivalence"
    SOLUTION_SET_EQUIVALENCE = "solution-set equivalence"
    SUBSTITUTION = "substitution"
    DIFFERENTIATION = "differentiation"
    BACKEND_IDENTITY = "backend identity"
    EXACT_ARITHMETIC = "exact arithmetic"


@dataclass(frozen=True, slots=True, kw_only=True)
class Verification:
    """Evidence that a transformation was checked successfully."""

    method: VerificationMethod
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepNote:
    """A labeled mathematical identity, rule, or substitution supporting a step."""

    label: str
    expression: Expression


@dataclass(frozen=True, slots=True, kw_only=True)
class SolutionStep:
    """One verified mathematical transformation."""

    rule: str
    before: Expression
    after: Expression
    explanation: str
    verification: Verification
    notes: tuple[StepNote, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarValue:
    """A scalar mathematical value."""

    expression: Expression


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceValue:
    """An ordered collection of mathematical values."""

    items: tuple[MathValue, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MappingEntry:
    """One key-value entry in a symbolic mapping."""

    key: Expression
    value: MathValue


@dataclass(frozen=True, slots=True, kw_only=True)
class MappingValue:
    """A deterministic symbolic mapping."""

    entries: tuple[MappingEntry, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixValue:
    """A rectangular matrix of scalar expressions."""

    rows: tuple[tuple[Expression, ...], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BooleanValue:
    """A mathematical truth value."""

    value: bool


type MathValue = ScalarValue | SequenceValue | MappingValue | MatrixValue | BooleanValue


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactResult:
    """A successfully solved exact query."""

    query: Query
    value: MathValue
    steps: tuple[SolutionStep, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class UnsolvedResult:
    """A valid query for which no verified answer was available."""

    query: Query
    reason: str
    steps: tuple[SolutionStep, ...]


type SolveResult = ExactResult | UnsolvedResult
