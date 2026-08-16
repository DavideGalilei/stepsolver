"""Human-readable elimination steps for small linear systems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendMathNote,
    BackendRowOperation,
    BackendSystem,
)
from stepsolver.results import VerificationMethod

_MIN_SYSTEM_SIZE = 2
_TWO_BY_TWO_SIZE = 2
type _Row = list[sp.Basic]


def _system(equations: Sequence[sp.Basic]) -> BackendSystem:
    return BackendSystem(equations=tuple(equations))


def _relations_display(relations: tuple[sp.Equality, ...]) -> sp.Equality | BackendSystem:
    if len(relations) == 1:
        return relations[0]
    return _system(relations)


def _linear_coefficients(
    equation: sp.Equality,
    variables: tuple[sp.Symbol, sp.Symbol],
) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
    difference = sp.expand(equation.lhs - equation.rhs)
    try:
        polynomial = sp.Poly(difference, variables)
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1:
        return None
    first, second = variables
    return (
        polynomial.coeff_monomial(first),
        polynomial.coeff_monomial(second),
        polynomial.coeff_monomial(1),
    )


def _solution_mapping(
    backend_value: object,
    variables: Sequence[sp.Symbol],
    *,
    require_every_variable: bool = True,
) -> Mapping[object, object] | None:
    if not isinstance(backend_value, Sequence) or isinstance(backend_value, str | bytes):
        return None
    solutions = cast("Sequence[object]", backend_value)
    if len(solutions) != 1 or not isinstance(solutions[0], Mapping):
        return None
    solution = cast("Mapping[object, object]", solutions[0])
    if require_every_variable and not all(
        isinstance(solution.get(variable), sp.Basic) for variable in variables
    ):
        return None
    if not all(
        isinstance(key, sp.Symbol) and isinstance(value, sp.Basic)
        for key, value in solution.items()
    ):
        return None
    return solution


def _solution_relations(
    solution: Mapping[object, object],
    variables: Sequence[sp.Symbol],
) -> tuple[sp.Equality, ...]:
    return tuple(
        sp.Eq(variable, value, evaluate=False)
        for variable in variables
        if isinstance((value := solution.get(variable)), sp.Basic)
    )


def _scaled_equation(equation: sp.Equality, multiplier: sp.Basic) -> sp.Equality:
    return sp.Eq(
        sp.expand(multiplier * equation.lhs),
        sp.expand(multiplier * equation.rhs),
        evaluate=False,
    )


def _opposite_multipliers(
    first_coefficient: sp.Basic,
    second_coefficient: sp.Basic,
) -> tuple[sp.Basic, sp.Basic]:
    if first_coefficient == sp.Integer(0):
        return sp.Integer(1), sp.Integer(0)
    if second_coefficient == sp.Integer(0):
        return sp.Integer(0), sp.Integer(1)
    common_factor = sp.gcd(first_coefficient, second_coefficient)
    if common_factor == sp.Integer(0):
        common_factor = sp.Integer(1)
    return (
        sp.simplify(second_coefficient / common_factor),
        sp.simplify(-first_coefficient / common_factor),
    )


def _dependent_two_by_two_steps(
    *,
    backend_value: object,
    variables: tuple[sp.Symbol, sp.Symbol],
    remaining_coefficient: sp.Basic,
    constant: sp.Basic,
    reduced_equation: sp.Equality,
    elimination_step: BackendDerivationStep,
) -> tuple[BackendDerivationStep, ...] | None:
    parametric_solution = _solution_mapping(
        backend_value,
        variables,
        require_every_variable=False,
    )
    if (
        parametric_solution is None
        or remaining_coefficient != sp.Integer(0)
        or constant != sp.Integer(0)
    ):
        return None
    relations = _solution_relations(parametric_solution, variables)
    if not relations:
        return ()
    free_variables = tuple(
        variable for variable in variables if variable not in parametric_solution
    )
    free_names = ", ".join(str(variable) for variable in free_variables)
    return (
        elimination_step,
        BackendDerivationStep(
            rule="Write the solution family",
            before=reduced_equation,
            after=_relations_display(relations),
            explanation=(
                "The last equation is an identity, so the system has infinitely many "
                f"solutions. Treat {free_names} as free and express the other variable "
                "in terms of it."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Substituting the displayed relation into either original equation "
                "produces an identity."
            ),
        ),
    )


def derive_two_by_two_linear_system(
    equations: tuple[sp.Equality, sp.Equality],
    variables: tuple[sp.Symbol, sp.Symbol],
    backend_value: object,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a two-equation linear system by elimination and back-substitution."""
    first_coefficients = _linear_coefficients(equations[0], variables)
    second_coefficients = _linear_coefficients(equations[1], variables)
    if first_coefficients is None or second_coefficients is None:
        return ()
    a1, b1, c1 = first_coefficients
    a2, b2, c2 = second_coefficients
    eliminate_index = 0 if a1 != sp.Integer(0) or a2 != sp.Integer(0) else 1
    if eliminate_index == 0:
        eliminated, remaining = variables
        multiplier_first, multiplier_second = _opposite_multipliers(a1, a2)
        remaining_coefficient = sp.simplify(multiplier_first * b1 + multiplier_second * b2)
        back_equation_index = 0 if a1 != sp.Integer(0) else 1
    else:
        remaining, eliminated = variables
        multiplier_first, multiplier_second = _opposite_multipliers(b1, b2)
        remaining_coefficient = sp.simplify(multiplier_first * a1 + multiplier_second * a2)
        back_equation_index = 0 if b1 != sp.Integer(0) else 1
    constant = sp.simplify(multiplier_first * c1 + multiplier_second * c2)
    reduced_equation = sp.Eq(remaining_coefficient * remaining, -constant, evaluate=False)
    equation_without_variable = multiplier_first == sp.Integer(
        0
    ) or multiplier_second == sp.Integer(0)
    elimination_step = BackendDerivationStep(
        rule=(
            f"Use the equation without {eliminated}"
            if equation_without_variable
            else f"Eliminate {eliminated}"
        ),
        before=_system(equations),
        after=reduced_equation,
        explanation=(
            f"One equation already contains no {eliminated}, so start with that equation."
            if equation_without_variable
            else (
                f"Scale the equations so the {eliminated}-coefficients are opposites, then "
                "add the equations."
            )
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "The displayed linear combination removes one variable without changing the "
            "system's solutions."
        ),
        notes=(
            ()
            if equation_without_variable
            else (
                BackendMathNote(
                    label="Scaled first equation",
                    expression=_scaled_equation(equations[0], multiplier_first),
                ),
                BackendMathNote(
                    label="Scaled second equation",
                    expression=_scaled_equation(equations[1], multiplier_second),
                ),
            )
        ),
    )
    solution = _solution_mapping(backend_value, variables)
    if solution is None:
        dependent_steps = _dependent_two_by_two_steps(
            backend_value=backend_value,
            variables=variables,
            remaining_coefficient=remaining_coefficient,
            constant=constant,
            reduced_equation=reduced_equation,
            elimination_step=elimination_step,
        )
        if dependent_steps is not None:
            return dependent_steps
        if (
            backend_value != []
            or remaining_coefficient != sp.Integer(0)
            or constant == sp.Integer(0)
        ):
            return ()
        return (
            elimination_step,
            BackendDerivationStep(
                rule="Conclude the system is inconsistent",
                before=reduced_equation,
                after=(),
                explanation="The elimination produced a false statement, so no pair can work.",
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The original system has an empty solution set.",
            ),
        )
    remaining_value = cast("sp.Basic", solution[remaining])
    remaining_relation = sp.Eq(remaining, remaining_value)
    back_equation = equations[back_equation_index]
    substituted_equation = sp.Eq(
        back_equation.lhs.subs(remaining, remaining_value),
        back_equation.rhs.subs(remaining, remaining_value),
        evaluate=False,
    )
    first_value = cast("sp.Basic", solution[variables[0]])
    second_value = cast("sp.Basic", solution[variables[1]])
    final_relations = (
        sp.Eq(variables[0], first_value),
        sp.Eq(variables[1], second_value),
    )
    steps = [elimination_step]
    if reduced_equation != remaining_relation:
        steps.append(
            BackendDerivationStep(
                rule=f"Solve for {remaining}",
                before=reduced_equation,
                after=remaining_relation,
                explanation=f"Divide by the coefficient of {remaining}.",
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The one-variable equation has the displayed exact solution.",
            )
        )
    steps.append(
        BackendDerivationStep(
            rule=f"Substitute back to find {eliminated}",
            before=_system((back_equation, remaining_relation)),
            after=_system(final_relations),
            explanation=(
                f"Insert {remaining} = {remaining_value} into equation "
                f"{back_equation_index + 1} and solve for {eliminated}."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Both values satisfy both original equations.",
            notes=(BackendMathNote(label="Substituted equation", expression=substituted_equation),),
        )
    )
    return tuple(steps)


def _linear_row(
    equation: sp.Equality,
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Basic, ...] | None:
    difference = sp.expand(equation.lhs - equation.rhs)
    try:
        polynomial = sp.Poly(difference, variables)
    except sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1:
        return None
    return (
        *(polynomial.coeff_monomial(variable) for variable in variables),
        -polynomial.coeff_monomial(1),
    )


def _row_equation(
    row: Sequence[sp.Basic],
    variables: tuple[sp.Symbol, ...],
) -> sp.Equality:
    left = sp.Add(
        *(coefficient * variable for coefficient, variable in zip(row, variables, strict=False))
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
    before = _system(_equations_from_rows(rows, variables))
    rows[pivot_row], rows[source] = rows[source], rows[pivot_row]
    return BackendDerivationStep(
        rule=f"Move an equation with {variable} into position",
        before=before,
        after=_system(_equations_from_rows(rows, variables)),
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
    before = _system(_equations_from_rows(rows, variables))
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
        after=_system(_equations_from_rows(rows, variables)),
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
    solution: Mapping[object, object],
    relations: tuple[sp.Equality, ...],
) -> BackendDerivationStep | None:
    free_variables = tuple(variable for variable in variables if variable not in solution)
    if not relations or not free_variables:
        return None
    free_names = ", ".join(str(variable) for variable in free_variables)
    return BackendDerivationStep(
        rule="Write the solution family",
        before=_system(_equations_from_rows(rows, variables)),
        after=_relations_display(relations),
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
    solution: Mapping[object, object],
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
        if not isinstance(value, sp.Basic):
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
                before=_system((equation, *relevant_relations)),
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
        after=_relations_display(relations),
        explanation=final_step.explanation,
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail="All displayed values satisfy every original equation.",
        notes=final_step.notes,
    )
    return tuple(steps)


def _derive_larger_linear_system(
    equations: tuple[sp.Equality, ...],
    variables: tuple[sp.Symbol, ...],
    backend_value: object,
) -> tuple[BackendDerivationStep, ...]:
    raw_rows = tuple(_linear_row(equation, variables) for equation in equations)
    if any(row is None for row in raw_rows):
        return ()
    rows = [list(cast("tuple[sp.Basic, ...]", row)) for row in raw_rows]
    rows, steps = _forward_elimination(rows, variables)
    contradiction = _contradiction_row(rows)
    if contradiction is not None:
        return (*steps, _contradiction_step(contradiction, variables))
    solution = _solution_mapping(backend_value, variables, require_every_variable=False)
    if solution is None:
        return ()
    relations = _solution_relations(solution, variables)
    if len(relations) != len(variables):
        parametric_step = _parametric_step(rows, variables, solution, relations)
        if parametric_step is None:
            return ()
        return (*steps, parametric_step)
    return (*steps, *_back_substitution_steps(rows, variables, solution, relations))


def derive_linear_system(
    equations: tuple[sp.Equality, ...],
    variables: tuple[sp.Symbol, ...],
    backend_value: object,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a linear system with elimination and back-substitution."""
    if len(equations) == _TWO_BY_TWO_SIZE and len(variables) == _TWO_BY_TWO_SIZE:
        return derive_two_by_two_linear_system(
            (equations[0], equations[1]),
            (variables[0], variables[1]),
            backend_value,
        )
    if len(equations) < _MIN_SYSTEM_SIZE or len(variables) < _MIN_SYSTEM_SIZE:
        return ()
    return _derive_larger_linear_system(equations, variables, backend_value)
