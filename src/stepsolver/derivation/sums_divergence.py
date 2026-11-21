"""Divergence tests for rational and radical summands."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendExpression,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNthRoot,
    BackendProduct,
    BackendSigma,
    BackendSum,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import contains_unevaluated_operation

_GEOMETRIC_POWER_ARITY = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class NthRootPowerMatch:
    """A positive base raised to and rooted by the summation index."""

    root_term: sp.Basic
    radicand: sp.Basic
    base: sp.Basic
    coefficient: sp.Basic


def derive_nth_term_divergence(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Check whether the summand tends to zero."""
    if upper != sp.oo or lower.is_integer is not True:
        return ()
    try:
        term_limit = sp.limit(expression, variable, sp.oo)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if contains_unevaluated_operation(term_limit):
        return ()
    is_nonzero = sp.simplify(sp.Ne(term_limit, sp.Integer(0)))
    if is_nonzero is not sp.true:
        return ()
    return (
        BackendDerivationStep(
            rule="Apply the nth-term divergence test",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation=(
                "A necessary condition for an infinite series to converge is that its terms "
                "approach zero. Here the term limit is nonzero, so the series diverges."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The summand's limit at infinity is not zero.",
            notes=(
                BackendMathNote(
                    label="Term limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=expression,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=term_limit,
                    ),
                ),
            ),
        ),
    )


def _match_nth_root_power(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> NthRootPowerMatch | None:
    """Find one term of the form ``(c * base**n)**(1/n)`` with positive factors."""
    for term in expression.as_ordered_terms():
        if not term.is_Pow or len(term.args) != _GEOMETRIC_POWER_ARITY:
            continue
        radicand, reciprocal_index = term.args
        if sp.simplify(reciprocal_index - 1 / variable) != sp.Integer(0):
            continue
        for factor in radicand.as_ordered_factors():
            if not factor.is_Pow or len(factor.args) != _GEOMETRIC_POWER_ARITY:
                continue
            base, exponent = factor.args
            coefficient = sp.simplify(radicand / factor)
            if (
                exponent == variable
                and base.is_positive is True
                and not coefficient.has(variable)
                and coefficient.is_positive is True
            ):
                return NthRootPowerMatch(
                    root_term=term,
                    radicand=radicand,
                    base=base,
                    coefficient=coefficient,
                )
    return None


def _nth_root_simplification(
    match: NthRootPowerMatch,
    variable: sp.Symbol,
) -> tuple[sp.Basic, BackendExpression]:
    """Return backend and human forms of ``base * root(c, n)``."""
    if match.coefficient == sp.Integer(1):
        return match.base, match.base
    reciprocal_index = sp.Pow(variable, -1, evaluate=False)
    coefficient_root = sp.Pow(match.coefficient, reciprocal_index, evaluate=False)
    backend_expression = sp.Mul(match.base, coefficient_root, evaluate=False)
    displayed_expression = BackendProduct(
        factors=(
            match.base,
            BackendNthRoot(radicand=match.coefficient, index=variable),
        ),
    )
    return backend_expression, displayed_expression


def _nth_term_divergence_conclusion(term_limit: sp.Basic) -> sp.Basic | None:
    """Choose the direction forced by an eventually signed nonzero term limit."""
    if term_limit == sp.oo or term_limit.is_positive is True:
        return sp.oo
    if term_limit == -sp.oo or term_limit.is_negative is True:
        return -sp.oo
    return None


def derive_nth_root_power_divergence(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    _result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Simplify an indexed root of a matching positive power before testing divergence."""
    if upper != sp.oo or lower.is_integer is not True or lower.is_positive is not True:
        return ()
    match = _match_nth_root_power(expression, variable)
    if match is None:
        return ()
    simplified_root, displayed_simplified_root = _nth_root_simplification(match, variable)
    remaining = sp.simplify(expression - match.root_term)
    simplified_expression = sp.simplify(simplified_root + remaining)
    try:
        term_limit = sp.limit(simplified_expression, variable, sp.oo)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if contains_unevaluated_operation(term_limit):
        return ()
    divergence_conclusion = _nth_term_divergence_conclusion(term_limit)
    if divergence_conclusion is None:
        return ()

    displayed_root = BackendNthRoot(radicand=match.radicand, index=variable)
    displayed_original: BackendExpression = displayed_root
    displayed_simplified = displayed_simplified_root
    if remaining != sp.Integer(0):
        displayed_original = BackendSum(terms=(displayed_root, remaining))
        displayed_simplified = BackendSum(terms=(displayed_simplified_root, remaining))
    original_sigma = BackendSigma(
        expression=displayed_original,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    simplified_sigma = BackendSigma(
        expression=displayed_simplified,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    return (
        BackendDerivationStep(
            rule="Simplify the indexed root",
            before=original_sigma,
            after=simplified_sigma,
            explanation=(
                "Separate the matching positive n-th power from the other positive "
                "factors. Its n-th root equals the base."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The indexed root and power have the same positive index.",
            notes=(
                BackendMathNote(
                    label="Root-power identity",
                    expression=BackendIdentity(
                        left=displayed_root,
                        right=displayed_simplified_root,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Apply the nth-term divergence test",
            before=simplified_sigma,
            after=divergence_conclusion,
            explanation=(
                "A convergent infinite series must have terms approaching zero. "
                "The simplified summand instead has a nonzero limit."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The simplified summand's limit at infinity is not zero.",
            notes=(
                BackendMathNote(
                    label="Term limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=displayed_simplified,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=term_limit,
                    ),
                ),
            ),
        ),
    )
