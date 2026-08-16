"""Runtime-neutral serialization for graphical StepSolver clients."""

from dataclasses import asdict, dataclass
from typing import Literal

from stepsolver.ast import OpaqueExpression
from stepsolver.formatter import format_ascii, format_expression
from stepsolver.latex import format_latex_expression, format_latex_value
from stepsolver.results import DivergenceKind, DivergentResult, ExactResult, SolveResult


@dataclass(frozen=True, slots=True, kw_only=True)
class StepNotePayload:
    """One labeled identity or substitution displayed with a step."""

    label: str
    expression_ascii: str
    expression_latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepPayload:
    """One browser-renderable verified solution step."""

    number: int
    rule: str
    explanation: str
    before_ascii: str
    after_ascii: str
    before_latex: str
    after_latex: str
    verification_method: str
    verification_detail: str
    notes: tuple[StepNotePayload, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SolvePayload:
    """Runtime-neutral representation of a completed solver request."""

    status: Literal["exact", "divergent", "unsolved"]
    source: str
    formatted_ascii: str
    result_latex: str | None
    reason: str | None
    steps: tuple[StepPayload, ...]

    def as_dict(self) -> dict[str, object]:
        """Convert this payload into standard JSON-compatible containers."""
        return asdict(self)


def _step_payloads(result: SolveResult) -> tuple[StepPayload, ...]:
    final_step_number = len(result.steps)
    return tuple(
        StepPayload(
            number=index,
            rule=step.rule,
            explanation=step.explanation,
            before_ascii=format_expression(step.before),
            after_ascii=format_expression(step.after),
            before_latex=format_latex_expression(step.before),
            after_latex=(
                format_latex_value(result.value)
                if (
                    isinstance(result, ExactResult)
                    and index == final_step_number
                    and isinstance(step.after, OpaqueExpression)
                )
                else format_latex_expression(step.after)
            ),
            verification_method=step.verification.method.value,
            verification_detail=step.verification.detail,
            notes=tuple(
                StepNotePayload(
                    label=note.label,
                    expression_ascii=format_expression(note.expression),
                    expression_latex=format_latex_expression(note.expression),
                )
                for note in step.notes
            ),
        )
        for index, step in enumerate(result.steps, start=1)
    )


def solve_payload(result: SolveResult) -> SolvePayload:
    """Render a solver result for either the HTTP or in-browser runtime."""
    steps = _step_payloads(result)
    if isinstance(result, ExactResult):
        return SolvePayload(
            status="exact",
            source=result.query.source,
            formatted_ascii=format_ascii(result),
            result_latex=format_latex_value(result.value),
            reason=None,
            steps=steps,
        )
    if isinstance(result, DivergentResult):
        divergence_latex = {
            DivergenceKind.POSITIVE_INFINITY: r"\text{Diverges to }+\infty",
            DivergenceKind.NEGATIVE_INFINITY: r"\text{Diverges to }-\infty",
            DivergenceKind.NONFINITE: r"\text{Does not converge}",
        }
        return SolvePayload(
            status="divergent",
            source=result.query.source,
            formatted_ascii=format_ascii(result),
            result_latex=divergence_latex[result.kind],
            reason=result.reason,
            steps=steps,
        )
    return SolvePayload(
        status="unsolved",
        source=result.query.source,
        formatted_ascii=format_ascii(result),
        result_latex=None,
        reason=result.reason,
        steps=steps,
    )
