"""Shared preparation and verification for polynomial equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isqrt
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendCrossedOut,
    BackendDerivationStep,
    BackendExpression,
    BackendIdentity,
    BackendIntroducedProduct,
    BackendNotEqual,
    BackendProduct,
    BackendQuotient,
    BackendStepConstraint,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import is_real_expression

_ROOT_VERIFICATION_DIGITS = 12
_DISPLAY_DIGITS = 7
_NEWTON_ITERATION_COUNT = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainRestrictions:
    """Original denominator restrictions retained through transformations."""

    denominators: tuple[sp.Basic, ...]
    excluded_values: tuple[sp.Basic, ...]
    displayed: tuple[BackendStepConstraint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedPolynomialEquation:
    """Normalized polynomial equation plus the steps needed to reach it."""

    current: sp.Equality
    expanded: sp.Basic
    restrictions: DomainRestrictions
    cleared_denominators: bool


def _is_basic_sequence(value: object) -> TypeGuard[Sequence[sp.Basic]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    items = cast("Sequence[object]", value)
    return all(isinstance(item, sp.Basic) for item in items)


def equivalent_step(
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
) -> BackendDerivationStep:
    """Build an equation step after verifying solution-set equivalence."""
    if solution_set(before, variable, excluded_values) != solution_set(
        after, variable, excluded_values
    ):
        message = "a proposed derivation step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "Both forms have the same solution set under the stated domain restrictions."
            if excluded_values
            else "Both forms have the same solution set for the target variable."
        ),
        introduced_constraints=introduced_constraints,
    )


def equivalent_display_step(
    *,
    rule: str,
    semantic_before: EquationBackendExpression,
    semantic_after: EquationBackendExpression,
    display_before: BackendExpression,
    display_after: BackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
    verification_detail: str = "Both displayed forms preserve the same solution set.",
) -> BackendDerivationStep:
    """Build a step with separate semantic and unsimplified displays."""
    if solution_set(semantic_before, variable, excluded_values) != solution_set(
        semantic_after, variable, excluded_values
    ):
        message = "a proposed display step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=display_before,
        after=display_after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=verification_detail,
        introduced_constraints=introduced_constraints,
    )


def solution_set(
    expression: EquationBackendExpression,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
) -> frozenset[str]:
    """Return a deterministic solution set after original exclusions."""
    equations = expression if isinstance(expression, tuple) else (expression,)
    roots: set[str] = set()
    for equation in equations:
        solved = sp.solve(equation, variable)
        if not _is_basic_sequence(solved):
            message = "equation verification did not produce a root sequence"
            raise TypeError(message)
        roots.update(
            (
                str(sp.simplify(root))
                if root.free_symbols
                else str(sp.N(root, _ROOT_VERIFICATION_DIGITS))
            )
            for root in solved
            if is_real_expression(root)
            if not any(
                sp.simplify(root - excluded) == sp.Integer(0) for excluded in excluded_values
            )
        )
    return frozenset(roots)


def relations(variable: sp.Symbol, roots: tuple[sp.Basic, ...]) -> tuple[sp.Basic, ...]:
    """Represent roots as explicit equations for display."""
    return tuple(sp.Eq(variable, root) for root in roots)


def _positive_divisors(value: int) -> tuple[int, ...]:
    small: list[int] = []
    large: list[int] = []
    for candidate in range(1, isqrt(value) + 1):
        if value % candidate != 0:
            continue
        small.append(candidate)
        paired = value // candidate
        if paired != candidate:
            large.append(paired)
    return (*small, *reversed(large))


def _rational_sort_key(value: sp.Rational) -> float:
    return float(str(value))


def rational_root_candidates(
    polynomial: sp.Poly,
    variable: sp.Symbol,
) -> tuple[sp.Rational, ...]:
    """Generate rational-root candidates in stable numerical order."""
    leading_coefficient = polynomial.coeff_monomial(variable ** polynomial.degree())
    constant_coefficient = polynomial.coeff_monomial(1)
    if not isinstance(leading_coefficient, sp.Integer) or not isinstance(
        constant_coefficient, sp.Integer
    ):
        return ()
    leading = abs(int(str(leading_coefficient)))
    constant = abs(int(str(constant_coefficient)))
    if leading == 0 or constant == 0:
        return ()
    candidates: set[sp.Rational] = {
        sp.Rational(sign * numerator, denominator)
        for numerator in _positive_divisors(constant)
        for denominator in _positive_divisors(leading)
        for sign in (-1, 1)
    }
    return tuple(sorted(candidates, key=_rational_sort_key))


def newton_values(
    expression: sp.Basic,
    variable: sp.Symbol,
    initial: sp.Basic,
) -> tuple[sp.Basic, ...]:
    """Compute a short, deterministic Newton iteration trace."""
    derivative = sp.diff(expression, variable)
    current = sp.N(initial, _DISPLAY_DIGITS)
    values: list[sp.Basic] = [current]
    for _index in range(_NEWTON_ITERATION_COUNT):
        slope = sp.N(derivative.subs(variable, current), _DISPLAY_DIGITS + 2)
        if slope == sp.Integer(0):
            break
        current = sp.N(
            current - expression.subs(variable, current) / slope,
            _DISPLAY_DIGITS,
        )
        values.append(current)
    return tuple(values)


def _domain_restrictions(
    denominator: sp.Basic,
    supplied_denominators: tuple[sp.Basic, ...],
    variable: sp.Symbol,
) -> DomainRestrictions:
    candidates: tuple[sp.Basic, ...] = supplied_denominators
    if not candidates:
        candidates = (denominator,)
    denominators: list[sp.Basic] = []
    for candidate in candidates:
        _constant_part, variable_part = candidate.as_independent(
            variable,
            as_Add=False,
        )
        variable_denominator = sp.factor(variable_part)
        if variable_denominator != sp.Integer(1):
            denominators.append(variable_denominator)
    unique_text = tuple(dict.fromkeys(str(item) for item in denominators))
    unique_denominators = tuple(
        next(item for item in denominators if str(item) == text) for text in unique_text
    )
    exclusions: list[sp.Basic] = []
    for domain_denominator in unique_denominators:
        solved = sp.solve(domain_denominator, variable)
        if _is_basic_sequence(solved):
            exclusions.extend(solved)
    excluded_values = tuple(dict.fromkeys(exclusions))
    denominator_constraints = tuple(
        BackendStepConstraint(
            explanation="An original denominator cannot equal zero.",
            expression=BackendNotEqual(left=item, right=sp.Integer(0)),
        )
        for item in unique_denominators
    )
    exclusion_constraints = tuple(
        BackendStepConstraint(
            explanation="This value is outside the domain of the original equation.",
            expression=BackendNotEqual(left=variable, right=value),
        )
        for value in excluded_values
    )
    return DomainRestrictions(
        denominators=unique_denominators,
        excluded_values=excluded_values,
        displayed=denominator_constraints + exclusion_constraints,
    )


def _common_denominator(
    denominator: sp.Basic,
    supplied_denominators: tuple[sp.Basic, ...],
) -> sp.Basic:
    """Find the least common multiplier from the equation's original fractions."""
    common: sp.Basic = sp.Integer(1)
    for candidate in (*supplied_denominators, denominator):
        common = sp.lcm(common, candidate)
    return sp.factor(common)


def append_if_changed(
    steps: list[BackendDerivationStep],
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
) -> EquationBackendExpression:
    """Append one verified transformation unless its display is unchanged."""
    if str(before) == str(after):
        return before
    steps.append(
        equivalent_step(
            rule=rule,
            before=before,
            after=after,
            explanation=explanation,
            variable=variable,
            excluded_values=excluded_values,
            introduced_constraints=introduced_constraints,
        )
    )
    return after


def _cleared_side(expression: sp.Basic, denominator: sp.Basic) -> sp.Basic:
    if expression == sp.Integer(0):
        return expression
    _numerator, expression_denominator = sp.fraction(sp.together(expression))
    product = expression * denominator
    if expression_denominator != sp.Integer(1):
        return sp.cancel(product)
    return sp.Mul(expression, denominator, evaluate=False)


def _multiplied_side(expression: sp.Basic, denominator: sp.Basic) -> sp.Basic:
    if expression == sp.Integer(0):
        return expression
    return sp.Mul(denominator, expression, evaluate=False)


def _cancelled_side(expression: sp.Basic, denominator: sp.Basic) -> BackendExpression:
    if expression == sp.Integer(0):
        return expression
    numerator, expression_denominator = sp.fraction(sp.together(expression))
    if expression_denominator == sp.Integer(1):
        return BackendProduct(factors=(denominator, expression))
    remaining_multiplier = sp.cancel(denominator / expression_denominator)
    numerator_factors: list[BackendExpression] = []
    if remaining_multiplier != sp.Integer(1):
        numerator_factors.append(remaining_multiplier)
    numerator_factors.append(BackendCrossedOut(expression=expression_denominator))
    if numerator != sp.Integer(1):
        numerator_factors.append(numerator)
    displayed_numerator: BackendExpression
    if len(numerator_factors) == 1:
        displayed_numerator = numerator_factors[0]
    else:
        displayed_numerator = BackendProduct(factors=tuple(numerator_factors))
    return BackendQuotient(
        numerator=displayed_numerator,
        denominator=BackendCrossedOut(expression=expression_denominator),
    )


def prepare_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    domain_denominators: tuple[sp.Basic, ...],
    steps: list[BackendDerivationStep],
) -> PreparedPolynomialEquation:
    """Clear original fractions and retain the equation used by later human steps."""
    difference = sp.together(equation.lhs - equation.rhs)
    numerator, denominator = sp.fraction(difference)
    clearing_denominator = _common_denominator(denominator, domain_denominators)
    restrictions = _domain_restrictions(denominator, domain_denominators, variable)
    if clearing_denominator == sp.Integer(1):
        return PreparedPolynomialEquation(
            current=equation,
            expanded=sp.expand(numerator),
            restrictions=restrictions,
            cleared_denominators=False,
        )

    multiplied = sp.Eq(
        _multiplied_side(equation.lhs, clearing_denominator),
        _multiplied_side(equation.rhs, clearing_denominator),
        evaluate=False,
    )
    multiplied_display = BackendIdentity(
        left=BackendIntroducedProduct(
            multiplier=clearing_denominator,
            expression=equation.lhs,
        ),
        right=BackendIntroducedProduct(
            multiplier=clearing_denominator,
            expression=equation.rhs,
        ),
    )
    cleared = sp.Eq(
        _cleared_side(equation.lhs, clearing_denominator),
        _cleared_side(equation.rhs, clearing_denominator),
        evaluate=False,
    )
    steps.append(
        equivalent_display_step(
            rule="Multiply both sides by the denominator",
            semantic_before=equation,
            semantic_after=multiplied,
            display_before=equation,
            display_after=multiplied_display,
            explanation=(
                "Multiply each side by the least common denominator so every fraction "
                "can be cleared."
            ),
            variable=variable,
            excluded_values=restrictions.excluded_values,
            introduced_constraints=restrictions.displayed,
            verification_detail=(
                "Multiplying both sides by the same nonzero expression preserves the solution set."
            ),
        )
    )
    cancellation_display = BackendIdentity(
        left=_cancelled_side(equation.lhs, clearing_denominator),
        right=_cancelled_side(equation.rhs, clearing_denominator),
    )
    steps.append(
        equivalent_display_step(
            rule="Cancel the common factors",
            semantic_before=multiplied,
            semantic_after=cleared,
            display_before=cancellation_display,
            display_after=cleared,
            explanation=(
                "Cancel each denominator with the matching nonzero factor introduced on "
                "the same side."
            ),
            variable=variable,
            excluded_values=restrictions.excluded_values,
            verification_detail="Canceling equal nonzero factors preserves the solution set.",
        )
    )
    expanded = sp.expand(
        numerator if clearing_denominator.has(variable) else cleared.lhs - cleared.rhs
    )
    return PreparedPolynomialEquation(
        current=cleared,
        expanded=expanded,
        restrictions=restrictions,
        cleared_denominators=True,
    )
