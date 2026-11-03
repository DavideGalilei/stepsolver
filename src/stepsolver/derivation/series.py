"""Taylor and Maclaurin derivations assembled from derivative evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Callable

_MAX_SERIES_TERMS = 12
_TEEN_MIN = 10
_TEEN_MAX = 20
_SSTR = cast("Callable[[object], str]", getattr(sp, "sstr"))
_DERIVATIVE = cast(
    "Callable[[sp.Basic, tuple[sp.Symbol, sp.Basic]], sp.Basic]",
    getattr(sp, "Derivative"),
)


def _truncated_polynomial(value: sp.Basic) -> sp.Basic | None:
    """Remove the order term from a backend series when one is present."""
    remove_order = getattr(value, "removeO", None)
    if callable(remove_order):
        polynomial = remove_order()
        if isinstance(polynomial, sp.Basic):
            return sp.expand(polynomial)
    return None


def _series_terms(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    order: int,
) -> tuple[tuple[int, sp.Basic, sp.Basic, sp.Basic], ...] | None:
    """Return ``(degree, derivative, value, coefficient)`` for each nonzero term."""
    terms: list[tuple[int, sp.Basic, sp.Basic, sp.Basic]] = []
    for degree in range(order):
        derivative = sp.diff(expression, variable, degree)
        value = sp.simplify(derivative.subs(variable, point))
        coefficient = sp.simplify(value / sp.factorial(sp.Integer(degree)))
        if coefficient == sp.Integer(0):
            continue
        if coefficient.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None
        terms.append((degree, derivative, value, coefficient))
    return tuple(terms)


def _ordinal(degree: int) -> str:
    if _TEEN_MIN <= degree % 100 <= _TEEN_MAX:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(degree % 10, "th")
    return f"{degree}{suffix}"


def _term_expression(coefficient: sp.Basic, offset: sp.Basic, degree: int) -> sp.Basic:
    if degree == 0:
        return coefficient
    if degree == 1:
        return coefficient * offset
    return coefficient * sp.Pow(offset, degree, evaluate=False)


def derive_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    order: int,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Expand a function through its derivative values at the expansion center."""
    if order < 1 or order > _MAX_SERIES_TERMS:
        return ()
    polynomial = _truncated_polynomial(result)
    if polynomial is None:
        return ()
    terms = _series_terms(expression, variable, point, order)
    if terms is None or not terms:
        return ()
    offset = variable - point
    built_terms = [
        _term_expression(coefficient, offset, degree) for degree, _, _, coefficient in terms
    ]
    rebuilt = sp.expand(sp.Add(*built_terms))
    if sp.simplify(rebuilt - polynomial) != sp.Integer(0):
        return ()

    index = sp.Symbol("k", integer=True, nonnegative=True)
    general_expression = (
        _DERIVATIVE(expression, (variable, index)) * sp.Pow(offset, index) / sp.factorial(index)
    )
    formula = BackendSigma(
        expression=general_expression,
        variable=index,
        lower=sp.Integer(0),
        upper=sp.Integer(order - 1),
    )
    center = "0" if point == sp.Integer(0) else _SSTR(point)
    if point == sp.Integer(0):
        expansion_rule = "Write the Maclaurin series formula"
    else:
        expansion_rule = f"Write the Taylor series about x = {center}"
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule=expansion_rule,
            before=expression,
            after=formula,
            explanation=(
                "Expand the function as a sum of its derivatives at the center, each "
                "multiplied by the corresponding power of the offset and divided by the "
                "matching factorial."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The Taylor formula expresses the truncated series through the derivative "
                "values at the chosen center."
            ),
        )
    ]
    assembled: list[sp.Basic] = []
    for degree, derivative, value, coefficient in terms:
        term = _term_expression(coefficient, offset, degree)
        assembled.append(term)
        steps.append(
            BackendDerivationStep(
                rule=f"Form the {_ordinal(degree)}-degree Taylor term",
                before=derivative,
                after=term,
                explanation=(
                    f"The {_ordinal(degree)} derivative evaluated at x = {center} is "
                    f"{_SSTR(value)}; dividing by {degree}! gives the coefficient "
                    f"{_SSTR(coefficient)}."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "The displayed derivative is exact, and its value at the center fixes "
                    "this Taylor coefficient."
                ),
                notes=(
                    BackendMathNote(
                        label="Derivative value at the center",
                        expression=BackendIdentity(
                            left=derivative.subs(variable, point),
                            right=value,
                        ),
                    ),
                ),
            )
        )
    assembled_expression = sp.Add(*assembled, evaluate=False)
    steps.append(
        BackendDerivationStep(
            rule="Assemble the truncated series",
            before=assembled_expression,
            after=result,
            explanation=(
                "Add the computed Taylor terms in increasing degree and record the order "
                "term that bounds the truncation error."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail=(
                "The assembled polynomial matches the backend truncation term by term."
            ),
        )
    )
    return tuple(steps)
