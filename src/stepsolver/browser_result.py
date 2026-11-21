"""Presentation records shared by the hosted and Pyodide solvers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from pydantic import ConfigDict


class _PayloadRecord:
    __slots__ = ()

    __pydantic_config__: ClassVar[ConfigDict] = {
        "extra": "forbid",
        "json_schema_serialization_defaults_required": True,
        "strict": True,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class ExplanationPartPayload(_PayloadRecord):
    """One prose or inline-math fragment in a browser explanation."""

    text: str | None = None
    latex: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class StepNotePayload(_PayloadRecord):
    """One labeled identity or substitution displayed with a step."""

    label: str
    expression_ascii: str
    expression_latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepConstraintPayload(_PayloadRecord):
    """One domain restriction introduced by a displayed step."""

    explanation: str
    expression_ascii: str
    expression_latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepPayload(_PayloadRecord):
    """One browser-renderable verified solution step."""

    number: int
    rule: str
    explanation: str
    explanation_parts: tuple[ExplanationPartPayload, ...]
    before_ascii: str
    after_ascii: str
    before_latex: str
    after_latex: str
    verification_method: str
    verification_detail: str
    notes: tuple[StepNotePayload, ...]
    introduced_constraints: tuple[StepConstraintPayload, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SolvePayload(_PayloadRecord):
    """Runtime-neutral representation of a completed solver request."""

    status: Literal["exact", "divergent", "undefined", "unsolved"]
    source: str
    formatted_ascii: str
    result_latex: str | None
    reason: str | None
    steps: tuple[StepPayload, ...]

    def as_dict(self) -> dict[str, object]:
        """Convert this payload into standard JSON-compatible containers."""
        return asdict(self)
