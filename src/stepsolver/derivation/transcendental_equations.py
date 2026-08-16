"""Human-readable derivations for supported transcendental equations."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod


def _is_exp_plus_variable(equation: sp.Equality, variable: sp.Symbol) -> bool:
    difference = sp.simplify(equation.lhs - equation.rhs)
    expected = sp.exp(variable) + variable
    return sp.simplify(difference - expected) == sp.Integer(0) or sp.simplify(
        difference + expected
    ) == sp.Integer(0)


def _has_lambert_solution(roots: tuple[sp.Basic, ...]) -> bool:
    expected = -sp.LambertW(1)
    return len(roots) == 1 and sp.simplify(roots[0] - expected) == sp.Integer(0)


def derive_transcendental_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive a supported transcendental equation using its standard inverse function."""
    if not _is_exp_plus_variable(equation, variable) or not _has_lambert_solution(roots):
        return ()
    substitution = sp.Symbol("u", real=True)
    normalized = sp.Eq(sp.exp(variable), -variable, evaluate=False)
    substituted = sp.Eq(sp.exp(-substitution), substitution, evaluate=False)
    lambert_form = sp.Eq(substitution * sp.exp(substitution), 1, evaluate=False)
    solved_substitution = sp.Eq(substitution, sp.LambertW(1), evaluate=False)
    final_relation = sp.Eq(variable, roots[0], evaluate=False)
    generic_value = sp.Symbol("z")
    generic_identity = BackendIdentity(
        left=sp.LambertW(generic_value) * sp.exp(sp.LambertW(generic_value)),
        right=generic_value,
    )
    return (
        BackendDerivationStep(
            rule="Isolate the exponential",
            before=equation,
            after=normalized,
            explanation="Move the variable term to the other side.",
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="Adding the same variable term to both sides is reversible.",
        ),
        BackendDerivationStep(
            rule="Substitute the negative variable",
            before=normalized,
            after=substituted,
            explanation=f"Let u = -{variable}. This puts the equation into an invertible form.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Replacing the variable by its stated substitution is reversible.",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=sp.Eq(substitution, -variable, evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Create the Lambert W pattern",
            before=substituted,
            after=lambert_form,
            explanation="Multiply both sides by e^u to obtain the pattern u e^u.",
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail=(
                "The exponential factor is never zero, so multiplication is reversible."
            ),
        ),
        BackendDerivationStep(
            rule="Apply the Lambert W function",
            before=lambert_form,
            after=solved_substitution,
            explanation="Lambert W is the inverse of the function that maps u to u e^u.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The defining identity of Lambert W gives the exact real value.",
            notes=(BackendMathNote(label="Lambert W identity", expression=generic_identity),),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=solved_substitution,
            after=final_relation,
            explanation=f"Since u = -{variable}, negate both sides to recover {variable}.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The exact value satisfies the original equation.",
        ),
    )
