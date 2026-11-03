"""Human-readable derivation for equations containing one square root."""

from __future__ import annotations

from typing import cast

import sympy as sp

from stepsolver.derivation.equations import derive_polynomial_equation
from stepsolver.derivation.equations_support import append_if_changed, relations
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod


def _square_root_powers(expression: sp.Basic) -> tuple[sp.Basic, ...]:
    """Find square-root powers because ``sqrt`` is represented as ``Pow(..., 1/2)``."""
    return tuple(power for power in expression.atoms(sp.Pow) if power.args[1] == sp.Rational(1, 2))


def _isolate_single_square_root(
    equation: sp.Equality,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic] | None:
    """Return the isolated radical and the other side, if the equation has one."""
    left_radicals = _square_root_powers(equation.lhs)
    right_radicals = _square_root_powers(equation.rhs)
    if bool(left_radicals) == bool(right_radicals):
        return None
    radical_side, other = (
        (equation.lhs, equation.rhs) if left_radicals else (equation.rhs, equation.lhs)
    )
    radicals = left_radicals or right_radicals
    if len(radicals) != 1:
        return None
    radical = radicals[0]
    if sp.simplify(radical_side - radical) != sp.Integer(0):
        return None
    radicand = radical.args[0]
    if not radicand.is_polynomial(variable) or not sp.expand(other).is_polynomial(variable):
        return None
    return radical, other


def _satisfies(equation: sp.Equality, variable: sp.Symbol, candidate: sp.Basic) -> bool:
    """Check a candidate in the original equation, before squaring."""
    residual = equation.lhs.subs(variable, candidate) - equation.rhs.subs(variable, candidate)
    return sp.simplify(residual) == sp.Integer(0)


def derive_radical_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive a square-root equation by squaring and rejecting extraneous roots."""
    isolated = _isolate_single_square_root(equation, variable)
    if isolated is None:
        return ()
    radical, other = isolated
    squared_left = sp.expand(radical**2)
    squared_right = sp.expand(other**2)
    squared_equation = sp.Eq(squared_left, squared_right, evaluate=False)
    normalized = sp.Eq(sp.expand(squared_left - squared_right), 0, evaluate=False)
    solved_candidates = tuple(dict.fromkeys(cast("list[sp.Basic]", sp.solve(normalized, variable))))
    if not solved_candidates:
        return ()

    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Square both sides",
            before=equation,
            after=squared_equation,
            explanation=(
                "Square both sides to remove the radical. Squaring can introduce "
                "candidates that do not satisfy the original equation, so every candidate "
                "will be checked."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "Both sides were squared exactly; this is a candidate-generating "
                "operation, not a solution-set equivalence."
            ),
            introduced_constraints=(
                BackendStepConstraint(
                    explanation="The square root requires a nonnegative radicand.",
                    expression=sp.Ge(radical.args[0], sp.Integer(0), evaluate=False),
                ),
            ),
        )
    ]
    append_if_changed(
        steps,
        rule="Expand and collect like terms",
        before=squared_equation,
        after=normalized,
        explanation="Expand both sides and move every term to one side.",
        variable=variable,
    )
    polynomial_steps = derive_polynomial_equation(
        normalized,
        variable,
        solved_candidates,
    )
    if polynomial_steps:
        steps.extend(polynomial_steps)
        candidate_before = polynomial_steps[-1].after
    else:
        candidate_before = relations(variable, solved_candidates)
        steps.append(
            BackendDerivationStep(
                rule="Solve the squared equation",
                before=normalized,
                after=candidate_before,
                explanation=(
                    "The squared equation is already linear, so solve it directly for the "
                    "candidate values."
                ),
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail=(
                    "The displayed candidate values satisfy the squared equation."
                ),
            )
        )

    checked_candidates = tuple(
        (candidate, _satisfies(equation, variable, candidate)) for candidate in solved_candidates
    )
    valid_candidates = tuple(candidate for candidate, satisfies in checked_candidates if satisfies)
    if {str(candidate) for candidate in roots} != {
        str(candidate) for candidate in valid_candidates
    }:
        return ()

    checks = tuple(
        BackendMathNote(
            label=(f"Test {variable} = {candidate} ({'satisfies' if satisfies else 'extraneous'})"),
            expression=BackendIdentity(
                left=equation.lhs.subs(variable, candidate),
                right=equation.rhs.subs(variable, candidate),
            ),
        )
        for candidate, satisfies in checked_candidates
    )
    steps.append(
        BackendDerivationStep(
            rule="Check candidates in the original equation",
            before=candidate_before,
            after=relations(variable, valid_candidates),
            explanation=(
                "Substitute each squared-equation candidate into the original equation. "
                "Keep only the candidates that make both sides equal."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Every candidate was substituted into the original equation; extraneous "
                "candidates were discarded."
            ),
            notes=checks,
        )
    )
    return tuple(steps)
