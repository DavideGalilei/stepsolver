"""Verify and render a differentiation rule."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendMathNote,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod


def verified_derivative_steps(
    *,
    rule: str,
    expression: sp.Basic,
    variable: sp.Symbol,
    raw_derivative: sp.Basic,
    result: sp.Basic,
    explanation: str,
    notes: tuple[BackendMathNote, ...],
    show_simplification: bool = False,
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
) -> tuple[BackendDerivationStep, ...]:
    """Verify a derivative before presenting its rule and optional simplification."""
    if sp.simplify(raw_derivative - result) != sp.Integer(0):
        message = "the derivative rule did not match the exact backend result"
        raise ValueError(message)
    steps = [
        BackendDerivationStep(
            rule=rule,
            before=BackendDerivative(expression=expression, variable=variable),
            after=raw_derivative,
            explanation=explanation,
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "The displayed rule was applied symbolically and checked against the exact "
                "derivative."
            ),
            notes=notes,
            introduced_constraints=introduced_constraints,
        )
    ]
    if show_simplification and str(raw_derivative) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Simplify the derivative",
                before=raw_derivative,
                after=result,
                explanation="Combine factors and like terms into a cleaner final form.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between both derivative forms gives zero."
                ),
            )
        )
    return tuple(steps)
