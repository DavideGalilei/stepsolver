"""Elimination and back-substitution for two linear equations."""

from __future__ import annotations

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
)
from stepsolver.results import VerificationMethod


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
    parametric_solution = solution_mapping(
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
    relations = solution_relations(parametric_solution, variables)
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
            after=relations_display(relations),
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
    first_coefficients = linear_row(equations[0], variables)
    second_coefficients = linear_row(equations[1], variables)
    if first_coefficients is None or second_coefficients is None:
        return ()
    a1, b1, rhs1 = first_coefficients
    a2, b2, rhs2 = second_coefficients
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
    constant = sp.simplify(-multiplier_first * rhs1 - multiplier_second * rhs2)
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
        before=system_display(equations),
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
    solution = solution_mapping(backend_value, variables)
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
    remaining_value = solution[remaining]
    remaining_relation = sp.Eq(remaining, remaining_value)
    back_equation = equations[back_equation_index]
    substituted_equation = sp.Eq(
        back_equation.lhs.subs(remaining, remaining_value),
        back_equation.rhs.subs(remaining, remaining_value),
        evaluate=False,
    )
    first_value = solution[variables[0]]
    second_value = solution[variables[1]]
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
            before=system_display((back_equation, remaining_relation)),
            after=system_display(final_relations),
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
