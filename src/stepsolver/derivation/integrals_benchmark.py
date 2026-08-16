"""Worked derivations for advanced substitution and identity integrals."""

from __future__ import annotations

from collections.abc import Callable

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendDifferential,
    BackendExpression,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

type _Strategy = Callable[[sp.Basic, sp.Symbol, sp.Basic], tuple[BackendDerivationStep, ...]]


def _matches(expression: sp.Basic, target: sp.Basic) -> bool:
    return sp.simplify(expression - target) == sp.Integer(0)


def _verified_final_step(
    *,
    rule: str,
    before: BackendExpression,
    result: sp.Basic,
    explanation: str,
) -> BackendDerivationStep:
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=result,
        explanation=explanation,
        verification_method=VerificationMethod.DIFFERENTIATION,
        verification_detail="Differentiating the result recovers the original integrand.",
    )


def _linear_tangent_square(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not _matches(integrand, variable * sp.tan(variable) ** 2):
        return ()
    rewritten = variable * (sp.sec(variable) ** 2 - 1)
    remaining = BackendDifference(
        left=BackendDifference(
            left=variable * sp.tan(variable),
            right=BackendIntegral(integrand=sp.tan(variable), variable=variable),
        ),
        right=variable**2 / 2,
    )
    return (
        BackendDerivationStep(
            rule="Rewrite the tangent square",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=rewritten, variable=variable),
            explanation="Use tan²(x) = sec²(x) - 1 so one term has an immediate derivative.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The Pythagorean identity makes the integrands equal.",
            notes=(
                BackendMathNote(
                    label="Identity",
                    expression=sp.Eq(
                        sp.tan(variable) ** 2,
                        sp.sec(variable) ** 2 - 1,
                        evaluate=False,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate the secant term by parts",
            before=BackendIntegral(integrand=rewritten, variable=variable),
            after=remaining,
            explanation=(
                "For ∫x sec²(x) dx, choose u = x and dv = sec²(x) dx; then integrate "
                "the remaining polynomial term."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The integration-by-parts identity was applied exactly.",
            notes=(
                BackendMathNote(label="u", expression=sp.Eq(sp.Symbol("u"), variable)),
                BackendMathNote(
                    label="dv",
                    expression=BackendDifferential(
                        variable=sp.Symbol("v"),
                        coefficient=sp.sec(variable) ** 2,
                    ),
                ),
            ),
        ),
        _verified_final_step(
            rule="Integrate the remaining tangent",
            before=remaining,
            result=result,
            explanation="Use ∫tan(x) dx = -ln|cos(x)| and collect the terms.",
        ),
    )


def _quadratic_over_square_root(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    radicand = variable**2 + 25
    if not _matches(integrand, variable**2 / sp.sqrt(radicand)):
        return ()
    rewritten = sp.sqrt(radicand) - 25 / sp.sqrt(radicand)
    return (
        BackendDerivationStep(
            rule="Split the numerator around the radicand",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=rewritten, variable=variable),
            explanation="Write x² as (x² + 25) - 25, then divide both terms by the square root.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the two terms reproduces the original fraction.",
            notes=(
                BackendMathNote(
                    label="Numerator rewrite",
                    expression=sp.Eq(variable**2, radicand - 25, evaluate=False),
                ),
            ),
        ),
        _verified_final_step(
            rule="Use the two standard square-root integrals",
            before=BackendIntegral(integrand=rewritten, variable=variable),
            result=result,
            explanation=(
                "Integrate √(x² + a²) and 1/√(x² + a²) with a = 5, then combine "
                "their logarithmic terms."
            ),
        ),
    )


def _power_square_root_substitution(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not _matches(integrand, variable**5 * sp.sqrt(2 - variable**3)):
        return ()
    substitution = sp.Symbol("u", real=True)
    transformed = -(2 - substitution) * sp.sqrt(substitution) / 3
    antiderivative_u = -sp.Rational(4, 9) * substitution ** sp.Rational(3, 2)
    antiderivative_u += sp.Rational(2, 15) * substitution ** sp.Rational(5, 2)
    return (
        BackendDerivationStep(
            rule="Substitute the expression under the square root",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=transformed, variable=substitution),
            explanation=("Let u = 2 - x³. Then x² dx = -du/3 and the remaining x³ becomes 2 - u."),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The differential and every remaining power were replaced exactly.",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=sp.Eq(substitution, 2 - variable**3, evaluate=False),
                ),
                BackendMathNote(
                    label="Differential",
                    expression=BackendDifferential(
                        variable=substitution,
                        coefficient=-3 * variable**2,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Expand into powers of u",
            before=BackendIntegral(integrand=transformed, variable=substitution),
            after=antiderivative_u,
            explanation="Distribute √u and apply the power rule to each term.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "Differentiating with respect to u returns the transformed integrand."
            ),
        ),
        _verified_final_step(
            rule="Substitute back",
            before=antiderivative_u,
            result=result,
            explanation="Replace u by 2 - x³ and simplify.",
        ),
    )


def _logarithmic_tangent_substitution(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    target = sp.tan(sp.log(variable)) ** 3 / variable
    if not _matches(integrand, target):
        return ()
    substitution = sp.Symbol("u", real=True)
    tangent_cube = sp.tan(substitution) ** 3
    rewritten = sp.tan(substitution) * (sp.sec(substitution) ** 2 - 1)
    return (
        BackendDerivationStep(
            rule="Substitute the logarithm",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=tangent_cube, variable=substitution),
            explanation="Let u = ln(x); the factor dx/x is exactly du.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Differentiating ln(x) gives the remaining 1/x factor.",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=sp.Eq(substitution, sp.log(variable), evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Rewrite the odd tangent power",
            before=BackendIntegral(integrand=tangent_cube, variable=substitution),
            after=BackendIntegral(integrand=rewritten, variable=substitution),
            explanation="Keep one tan(u) factor and replace tan²(u) by sec²(u) - 1.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The Pythagorean identity preserves the integrand.",
        ),
        _verified_final_step(
            rule="Integrate and substitute back",
            before=BackendIntegral(integrand=rewritten, variable=substitution),
            result=result,
            explanation=(
                "Use w = tan(u) for the sec² term, integrate the remaining tan(u), then "
                "replace u by ln(x)."
            ),
        ),
    )


def derive_advanced_substitution_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Choose a dedicated human method for a supported advanced integral."""
    strategies: tuple[_Strategy, ...] = (
        _linear_tangent_square,
        _quadratic_over_square_root,
        _power_square_root_substitution,
        _logarithmic_tangent_substitution,
    )
    for strategy in strategies:
        steps = strategy(integrand, variable, result)
        if steps:
            return steps
    return ()
