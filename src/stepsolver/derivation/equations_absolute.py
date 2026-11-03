"""Human-readable derivations for absolute-value equations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendStepConstraint,
    BackendSystem,
)
from stepsolver.results import VerificationMethod


def _split_absolute_value(equation: sp.Equality) -> tuple[sp.Basic, sp.Basic] | None:
    """Return ``(argument, other_side)`` for ``|argument| = other_side``."""
    left_absolute = equation.lhs if equation.lhs.func == sp.Abs else None
    right_absolute = equation.rhs if equation.rhs.func == sp.Abs else None
    if (left_absolute is None) == (right_absolute is None):
        return None
    absolute, other = (
        (left_absolute, equation.rhs)
        if left_absolute is not None
        else (right_absolute, equation.lhs)
    )
    if absolute is None or other.has(sp.Abs):
        return None
    return absolute.args[0], other


def _solve_branch(equation: sp.Equality, variable: sp.Symbol) -> tuple[sp.Basic, ...] | None:
    """Solve one sign branch, rejecting branches with infinitely many solutions."""
    difference = sp.simplify(equation.lhs - equation.rhs)
    if difference == sp.Integer(0):
        return None
    try:
        solved = sp.solve(equation, variable)
    except (NotImplementedError, TypeError, ValueError):
        return None
    if not isinstance(solved, Sequence) or isinstance(solved, str | bytes) or not solved:
        return None
    candidates = cast("Sequence[object]", solved)
    if not all(isinstance(root, sp.Basic) for root in candidates):
        return None
    return tuple(dict.fromkeys(cast("Sequence[sp.Basic]", candidates)))


def _satisfies(equation: sp.Equality, variable: sp.Symbol, candidate: sp.Basic) -> bool:
    """Check a candidate in the original absolute-value equation."""
    residual = equation.lhs.subs(variable, candidate) - equation.rhs.subs(variable, candidate)
    try:
        return sp.simplify(residual) == sp.Integer(0)
    except (TypeError, ValueError):
        return False


def derive_absolute_value_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive ``|A| = B`` by splitting into two sign branches and checking them."""
    split = _split_absolute_value(equation)
    if split is None:
        return ()
    argument, other = split
    branches = (
        sp.Eq(argument, other, evaluate=False),
        sp.Eq(argument, -other, evaluate=False),
    )
    candidates: list[sp.Basic] = []
    for branch in branches:
        branch_roots = _solve_branch(branch, variable)
        if branch_roots is None:
            return ()
        candidates.extend(branch_roots)
    unique_candidates = tuple(dict.fromkeys(candidates))
    checked = tuple(
        (candidate, _satisfies(equation, variable, candidate)) for candidate in unique_candidates
    )
    valid = tuple(candidate for candidate, satisfies in checked if satisfies)
    if {str(sp.simplify(root)) for root in roots} != {
        str(sp.simplify(candidate)) for candidate in valid
    }:
        return ()

    display = BackendSystem(equations=branches)
    candidate_relations = tuple(
        sp.Eq(variable, candidate, evaluate=False) for candidate in unique_candidates
    )
    checks = tuple(
        BackendMathNote(
            label=f"Test {variable} = {candidate} ({'satisfies' if satisfies else 'extraneous'})",
            expression=BackendIdentity(
                left=equation.lhs.subs(variable, candidate),
                right=equation.rhs.subs(variable, candidate),
            ),
        )
        for candidate, satisfies in checked
    )
    return (
        BackendDerivationStep(
            rule="Split into the two absolute-value branches",
            before=equation,
            after=display,
            explanation=(
                "The absolute value of an expression equals a nonnegative number when the "
                "expression equals that number or its negative. Solve both sign branches, then "
                "keep only the candidates that satisfy the original equation."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail=(
                "Together the two sign branches cover every value whose absolute value is the "
                "right-hand side; extraneous candidates are removed afterwards."
            ),
            notes=(
                BackendMathNote(
                    label="Absolute-value definition",
                    expression=BackendIdentity(
                        left=sp.Abs(argument),
                        right=BackendSystem(
                            equations=(argument, -argument),
                        ),
                    ),
                ),
            ),
            introduced_constraints=(
                BackendStepConstraint(
                    explanation="An absolute value is nonnegative, so the other side must be too.",
                    expression=sp.Ge(other, sp.Integer(0), evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Solve both sign branches",
            before=display,
            after=candidate_relations,
            explanation=(
                "Solve the branch with the positive right-hand side and the branch with its "
                "negative separately, then combine the candidate values."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail=(
                "The candidate list is the union of the exact solutions of the two branches."
            ),
        ),
        BackendDerivationStep(
            rule="Check candidates in the original equation",
            before=candidate_relations,
            after=tuple(sp.Eq(variable, candidate, evaluate=False) for candidate in valid),
            explanation=(
                "Substitute each branch candidate into the original absolute-value equation and "
                "discard any candidate for which the two sides disagree."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Every remaining candidate was substituted into the original equation."
            ),
            notes=checks,
        ),
    )
