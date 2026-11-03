"""Student-facing derivations for numeric nested square roots."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendRelation,
)
from stepsolver.results import VerificationMethod

_SQUARE_ROOT = sp.Rational(1, 2)
_POWER_ARITY = 2


def _is_numeric_square_root(expression: sp.Basic) -> bool:
    """Return whether the expression is a symbol-free square root."""
    return (
        not expression.free_symbols
        and expression.func is sp.Pow
        and len(expression.args) == _POWER_ARITY
        and expression.args[1] == _SQUARE_ROOT
    )


def derive_radical_denesting(
    expression: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Explain a numeric nested square root that was rewritten as a sum."""
    if not _is_numeric_square_root(expression) or result.free_symbols:
        return ()
    if result.func is not sp.Add or result.is_positive is not True:
        return ()
    radicand = expression.args[0]
    if sp.simplify(sp.Pow(result, 2) - radicand) != sp.Integer(0):
        return ()
    inner_square = sp.Pow(result, 2, evaluate=False)
    first_after = sp.Pow(inner_square, sp.Rational(1, 2), evaluate=False)
    return (
        BackendDerivationStep(
            rule="Rewrite the radicand as a perfect square",
            before=expression,
            after=first_after,
            explanation=(
                "Find a positive quantity whose square is the radicand, then write the radicand "
                "as that perfect square."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="Squaring the proposed quantity recovers the radicand.",
            notes=(
                BackendMathNote(
                    label="Perfect square",
                    expression=BackendIdentity(left=radicand, right=inner_square),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Take the positive square root",
            before=first_after,
            after=result,
            explanation=(
                "The rewritten quantity is positive, so the principal square root is the "
                "quantity itself rather than its negative."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The proposed value is positive and squaring it recovers the radicand."
            ),
            notes=(
                BackendMathNote(
                    label="Positive root",
                    expression=BackendRelation(operator=">", left=result, right=sp.Integer(0)),
                ),
            ),
        ),
    )
