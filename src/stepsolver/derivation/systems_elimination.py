"""Row operations and back-substitution for larger systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.derivation.linear_system_data import (
    linear_row,
    relations_display,
    solution_mapping,
    solution_relations,
    system_display,
)
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendMathNote,
    BackendRowOperation,
)
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type _Row = list[sp.Basic]


def _row_equation(
    row: Sequence[sp.Basic],
    variables: tuple[sp.Symbol, ...],
) -> sp.Equality:
    left = sp.Add(
        *(coefficient * variable for coefficient, variable in zip(row[:-1], variables, strict=True))
    )
    return sp.Eq(left, row[-1], evaluate=False)


def _equations_from_rows(
    rows: Sequence[Sequence[sp.Basic]],
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Equality, ...]:
    return tuple(_row_equation(row, variables) for row in rows)


def _first_nonzero_coefficient(row: Sequence[sp.Basic]) -> int | None:
    return next(
        (index for index, coefficient in enumerate(row[:-1]) if coefficient != sp.Integer(0)),
        None,
    )


def _row_operation_explanation(factor: sp.Basic, pivot: int, target: int) -> str:
    if factor.could_extract_minus_sign():
        return f"Add {-factor} times equation {pivot + 1} to equation {target + 1}."
    return f"Subtract {factor} times equation {pivot + 1} from equation {target + 1}."


def _swap_step(
    rows: list[_Row],
    variables: tuple[sp.Symbol, ...],
    pivot_row: int,
    source: int,
    variable: sp.Symbol,
) -> BackendDerivationStep:
    before = system_display(_equations_from_rows(rows, variables))
    rows[pivot_row], rows[source] = rows[source], rows[pivot_row]
    return BackendDerivationStep(
        rule=f"Move an equation with {variable} into position",
        before=before,
        after=system_display(_equations_from_rows(rows, variables)),
        explanation=(
            f"Swap equations {pivot_row + 1} and {source + 1} so the next elimination "
            f"step has a nonzero {variable}-coefficient."
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail="Reordering equations does not change their common solutions.",
    )


def _replace_row_step(
    rows: list[_Row],
    variables: tuple[sp.Symbol, ...],
    pivot_row: int,
    target: int,
    column: int,
) -> BackendDerivationStep:
    before = system_display(_equations_from_rows(rows, variables))
    factor = sp.simplify(rows[target][column] / rows[pivot_row][column])
    rows[target] = [
        sp.simplify(target_value - factor * pivot_value)
        for target_value, pivot_value in zip(rows[target], rows[pivot_row], strict=True)
    ]
    dependent = all(value == sp.Integer(0) for value in rows[target])
    variable = variables[column]
    rule = (
        "Recognize a dependent equation"
        if dependent
        else f"Eliminate {variable} from equation {target + 1}"
    )
    explanation = (
        "This row reduces to an identity, so it repeats information already in the other equations."
        if dependent
        else _row_operation_explanation(factor, pivot_row, target)
    )
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=system_display(_equations_from_rows(rows, variables)),
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "Replacing one equation by itself minus a multiple of another preserves the "
            "system's solution set."
        ),
        notes=(
            BackendMathNote(
                label="Row operation",
                expression=BackendRowOperation(
                    target=target + 1,
                    source=pivot_row + 1,
                    factor=factor,
                ),
            ),
        ),
    )


def _forward_elimination(
    rows: list[_Row],
    variables: tuple[sp.Symbol, ...],
) -> tuple[list[_Row], list[BackendDerivationStep]]:
    steps: list[BackendDerivationStep] = []
    pivot_row = 0
    for column, variable in enumerate(variables):
        source = next(
            (
                index
                for index in range(pivot_row, len(rows))
                if rows[index][column] != sp.Integer(0)
            ),
            None,
        )
        if source is None:
            continue
        if source != pivot_row:
            steps.append(_swap_step(rows, variables, pivot_row, source, variable))
        steps.extend(
            [
                _replace_row_step(rows, variables, pivot_row, target, column)
                for target in range(pivot_row + 1, len(rows))
                if rows[target][column] != sp.Integer(0)
            ]
        )
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return rows, steps


def _contradiction_row(rows: Sequence[_Row]) -> _Row | None:
    return next(
        (
            row
            for row in rows
            if _first_nonzero_coefficient(row) is None and sp.simplify(row[-1]) != sp.Integer(0)
        ),
        None,
    )


def _contradiction_step(
    row: _Row,
    variables: tuple[sp.Symbol, ...],
) -> BackendDerivationStep:
    return BackendDerivationStep(
        rule="Conclude the system is inconsistent",
        before=_row_equation(row, variables),
        after=(),
        explanation="A row says that zero equals a nonzero number, which is impossible.",
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail="No values can satisfy the contradictory row.",
    )


def _parametric_step(
    rows: Sequence[_Row],
    variables: tuple[sp.Symbol, ...],
    solution: Mapping[sp.Symbol, sp.Basic],
    relations: tuple[sp.Equality, ...],
) -> BackendDerivationStep | None:
    free_variables = tuple(variable for variable in variables if variable not in solution)
    if not relations or not free_variables:
        return None
    free_names = ", ".join(str(variable) for variable in free_variables)
    return BackendDerivationStep(
        rule="Write the solution family",
        before=system_display(_equations_from_rows(rows, variables)),
        after=relations_display(relations),
        explanation=(
            f"There is no pivot for {free_names}, so choose it freely and express the pivot "
            "variables in terms of it."
        ),
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail="The parametric relations satisfy every original equation.",
    )


def _back_substitution_steps(
    rows: Sequence[_Row],
    variables: tuple[sp.Symbol, ...],
    solution: Mapping[sp.Symbol, sp.Basic],
    relations: tuple[sp.Equality, ...],
) -> tuple[BackendDerivationStep, ...]:
    known_relations: list[sp.Equality] = []
    steps: list[BackendDerivationStep] = []
    for row in reversed(rows):
        column = _first_nonzero_coefficient(row)
        if column is None:
            continue
        variable = variables[column]
        value = solution.get(variable)
        if value is None:
            return ()
        relation = sp.Eq(variable, value, evaluate=False)
        equation = _row_equation(row, variables)
        relevant_relations = tuple(
            known
            for known in known_relations
            if isinstance(known.lhs, sp.Symbol) and known.lhs in equation.free_symbols
        )
        steps.append(
            BackendDerivationStep(
                rule=(
                    f"Substitute back for {variable}"
                    if relevant_relations
                    else f"Solve for {variable}"
                ),
                before=system_display((equation, *relevant_relations)),
                after=relation,
                explanation=(
                    f"Use the values already found in this equation, then isolate {variable}."
                    if relevant_relations
                    else f"The last equation contains only {variable}; isolate it."
                ),
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The displayed value solves this row of the system.",
            )
        )
        known_relations.insert(0, relation)
    if not steps:
        return ()
    final_step = steps[-1]
    steps[-1] = BackendDerivationStep(
        rule=final_step.rule,
        before=final_step.before,
        after=relations_display(relations),
        explanation=final_step.explanation,
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail="All displayed values satisfy every original equation.",
        notes=final_step.notes,
    )
    return tuple(steps)


def derive_larger_linear_system(
    equations: tuple[sp.Equality, ...],
    variables: tuple[sp.Symbol, ...],
    backend_value: object,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the solution using elementary row operations."""
    rows: list[_Row] = []
    for equation in equations:
        row = linear_row(equation, variables)
        if row is None:
            return ()
        rows.append(list(row))
    rows, steps = _forward_elimination(rows, variables)
    contradiction = _contradiction_row(rows)
    if contradiction is not None:
        return (*steps, _contradiction_step(contradiction, variables))
    solution = solution_mapping(backend_value, variables, require_every_variable=False)
    if solution is None:
        return ()
    relations = solution_relations(solution, variables)
    if len(relations) != len(variables):
        parametric_step = _parametric_step(rows, variables, solution, relations)
        if parametric_step is None:
            return ()
        return (*steps, parametric_step)
    return (*steps, *_back_substitution_steps(rows, variables, solution, relations))
