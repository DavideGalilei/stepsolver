"""Public polynomial-equation derivation dispatcher."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.equations_linear import derive_linear_steps
from stepsolver.derivation.equations_polynomial import (
    derive_cubic_steps,
    derive_generic_polynomial_steps,
    derive_quadratic_steps,
)
from stepsolver.derivation.equations_support import append_if_changed, prepare_polynomial_equation
from stepsolver.derivation.model import (
    BackendDerivationStep,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod

_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2
_CUBIC_DEGREE = 3
_CONSTANT_DEGREE = 0


def derive_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
    domain_denominators: tuple[sp.Basic, ...] = (),
) -> tuple[BackendDerivationStep, ...]:
    """Derive detailed steps for rational, linear, and quadratic equations."""
    steps: list[BackendDerivationStep] = []
    prepared = prepare_polynomial_equation(
        equation,
        variable,
        domain_denominators,
        steps,
    )
    restrictions = prepared.restrictions
    excluded_values = restrictions.excluded_values
    current: EquationBackendExpression = prepared.current
    expanded = prepared.expanded
    normalized = sp.Eq(expanded, 0, evaluate=False)
    polynomial = sp.Poly(expanded, variable)
    degree = polynomial.degree()
    if prepared.cleared_denominators and degree != _LINEAR_DEGREE:
        current = append_if_changed(
            steps,
            rule="Expand and collect like terms",
            before=current,
            after=normalized,
            explanation="Expand the products, move every term to one side, and combine terms.",
            variable=variable,
            excluded_values=excluded_values,
        )
    if degree == _CONSTANT_DEGREE:
        current = append_if_changed(
            steps,
            rule="Simplify the equation",
            before=current,
            after=normalized,
            explanation="Combine like terms on both sides.",
            variable=variable,
            excluded_values=excluded_values,
        )
        steps.append(
            BackendDerivationStep(
                rule="Conclude there are no solutions",
                before=current,
                after=(),
                explanation="A nonzero constant cannot equal zero.",
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The simplified contradiction has an empty solution set.",
            )
        )
        return tuple(steps)
    if degree == _LINEAR_DEGREE:
        detail = derive_linear_steps(
            prepared.current,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    elif degree == _QUADRATIC_DEGREE:
        normalized_current = normalized
        if not steps:
            if sp.simplify(equation.rhs) == sp.Integer(0):
                normalized_current = equation
            else:
                current = append_if_changed(
                    steps,
                    rule="Write in standard form",
                    before=current,
                    after=normalized,
                    explanation="Move every term to one side and combine like terms.",
                    variable=variable,
                    excluded_values=excluded_values,
                )
                if isinstance(current, sp.Equality):
                    normalized_current = current
        detail = derive_quadratic_steps(
            normalized_current,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    elif degree == _CUBIC_DEGREE:
        detail = derive_cubic_steps(
            normalized if steps else equation,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    else:
        detail = derive_generic_polynomial_steps(
            normalized if steps else equation,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    steps.extend(detail)
    return tuple(steps)
