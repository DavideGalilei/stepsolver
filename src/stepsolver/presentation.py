"""Runtime-neutral serialization for graphical StepSolver clients."""

from dataclasses import asdict, dataclass
from typing import Literal

from stepsolver.ast import FunctionCall, OpaqueExpression, Operation, Relation
from stepsolver.formatter import format_ascii, format_expression
from stepsolver.latex import format_latex_expression, format_latex_value
from stepsolver.results import (
    DivergenceKind,
    DivergentResult,
    ExactResult,
    MappingValue,
    MathValue,
    SequenceValue,
    SolveResult,
    UndefinedResult,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StepNotePayload:
    """One labeled identity or substitution displayed with a step."""

    label: str
    expression_ascii: str
    expression_latex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class StepConstraintPayload:
    """One domain restriction introduced by a displayed step."""

    explanation: str
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
    introduced_constraints: tuple[StepConstraintPayload, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SolvePayload:
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


def _single_variable_solutions_latex(value: MathValue) -> str | None:
    if not isinstance(value, SequenceValue) or not value.items:
        return None
    mappings: list[MappingValue] = []
    for item in value.items:
        if not isinstance(item, MappingValue) or len(item.entries) != 1:
            return None
        mappings.append(item)
    variable = mappings[0].entries[0].key
    if any(mapping.entries[0].key != variable for mapping in mappings[1:]):
        return None
    variable_latex = format_latex_expression(variable)
    equations = (
        rf"{variable_latex} = {format_latex_value(mapping.entries[0].value)}"
        for mapping in mappings
    )
    return r"\quad\text{or}\quad ".join(equations)


def _exact_result_latex(result: ExactResult) -> str:
    if result.query.operation is Operation.SOLVE:
        if result.steps:
            final_step = result.steps[-1]
            if (
                isinstance(final_step.after, FunctionCall)
                and str(final_step.after.name) == "approximate_solutions"
            ):
                return format_latex_expression(final_step.after)
            if (
                isinstance(final_step.after, FunctionCall)
                and str(final_step.after.name) == "cardano_solution"
            ):
                exact = format_latex_expression(final_step.after)
                decimal_note = next(
                    (note for note in final_step.notes if note.label == "Decimal check"),
                    None,
                )
                if decimal_note is not None and isinstance(decimal_note.expression, Relation):
                    approximation = format_latex_expression(decimal_note.expression.right)
                    variable = format_latex_expression(decimal_note.expression.left)
                    return rf"{exact}\quad\left({variable} \approx {approximation}\right)"
                return exact
        solutions = _single_variable_solutions_latex(result.value)
        if solutions is not None:
            return solutions
    return format_latex_value(result.value)


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
                _exact_result_latex(result)
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
            introduced_constraints=tuple(
                StepConstraintPayload(
                    explanation=constraint.explanation,
                    expression_ascii=format_expression(constraint.expression),
                    expression_latex=format_latex_expression(constraint.expression),
                )
                for constraint in step.introduced_constraints
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
            result_latex=_exact_result_latex(result),
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
    if isinstance(result, UndefinedResult):
        return SolvePayload(
            status="undefined",
            source=result.query.source,
            formatted_ascii=format_ascii(result),
            result_latex=r"\text{Undefined}",
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
