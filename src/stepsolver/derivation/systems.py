"""Human-readable elimination steps for small linear systems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import sympy as sp

from stepsolver.derivation.model import BackendDerivationStep, BackendMathNote
from stepsolver.results import VerificationMethod


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
    variables: tuple[sp.Symbol, sp.Symbol],
) -> Mapping[object, object] | None:
    if not isinstance(backend_value, Sequence) or isinstance(backend_value, str | bytes):
        return None
    solutions = cast("Sequence[object]", backend_value)
    if len(solutions) != 1 or not isinstance(solutions[0], Mapping):
        return None
    solution = cast("Mapping[object, object]", solutions[0])
    if not all(isinstance(solution.get(variable), sp.Basic) for variable in variables):
        return None
    return solution


def _scaled_equation(equation: sp.Equality, multiplier: sp.Basic) -> sp.Equality:
    return sp.Eq(
        sp.expand(multiplier * equation.lhs),
        sp.expand(multiplier * equation.rhs),
        evaluate=False,
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
        multiplier_first, multiplier_second = a2, -a1
        remaining_coefficient = sp.simplify(multiplier_first * b1 + multiplier_second * b2)
    else:
        remaining, eliminated = variables
        multiplier_first, multiplier_second = b2, -b1
        remaining_coefficient = sp.simplify(multiplier_first * a1 + multiplier_second * a2)
    constant = sp.simplify(multiplier_first * c1 + multiplier_second * c2)
    reduced_equation = sp.Eq(remaining_coefficient * remaining, -constant, evaluate=False)
    elimination_step = BackendDerivationStep(
        rule=f"Eliminate {eliminated}",
        before=equations,
        after=reduced_equation,
        explanation=(
            f"Scale the equations so the {eliminated}-coefficients are opposites, then add "
            "the equations."
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "The displayed linear combination removes one variable without changing the "
            "system's solutions."
        ),
        notes=(
            BackendMathNote(
                label="Scaled first equation",
                expression=_scaled_equation(equations[0], multiplier_first),
            ),
            BackendMathNote(
                label="Scaled second equation",
                expression=_scaled_equation(equations[1], multiplier_second),
            ),
        ),
    )
    solution = _solution_mapping(backend_value, variables)
    if solution is None:
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
    eliminated_value = solution[eliminated]
    if not isinstance(remaining_value, sp.Basic) or not isinstance(eliminated_value, sp.Basic):
        return ()
    remaining_relation = sp.Eq(remaining, remaining_value)
    substituted_equation = sp.Eq(
        equations[0].lhs.subs(remaining, remaining_value),
        equations[0].rhs.subs(remaining, remaining_value),
        evaluate=False,
    )
    first_value = solution[variables[0]]
    second_value = solution[variables[1]]
    if not isinstance(first_value, sp.Basic) or not isinstance(second_value, sp.Basic):
        return ()
    final_relations = (
        sp.Eq(variables[0], first_value),
        sp.Eq(variables[1], second_value),
    )
    return (
        elimination_step,
        BackendDerivationStep(
            rule=f"Solve for {remaining}",
            before=reduced_equation,
            after=remaining_relation,
            explanation=f"Divide by the coefficient of {remaining}.",
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The one-variable equation has the displayed exact solution.",
        ),
        BackendDerivationStep(
            rule=f"Substitute back to find {eliminated}",
            before=(equations[0], remaining_relation),
            after=final_relations,
            explanation=(
                f"Insert {remaining} = {remaining_value} into the first equation and solve "
                f"for {eliminated}."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Both values satisfy both original equations.",
            notes=(BackendMathNote(label="Substituted equation", expression=substituted_equation),),
        ),
    )
