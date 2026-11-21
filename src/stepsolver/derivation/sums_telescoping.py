"""Telescoping with partial fractions."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

_SQUARE_POWER = 2


def _has_constant_numerator_quadratic_denominator(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> bool:
    """Return whether a term can have two shifted linear denominator factors."""
    numerator, denominator = sp.fraction(sp.cancel(expression))
    if numerator.has(variable):
        return False
    try:
        denominator_polynomial = sp.Poly(denominator, variable)
    except sp.PolynomialError:
        return False
    return denominator_polynomial.degree() == _SQUARE_POWER


def derive_telescoping_rational_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use partial fractions for c/(n(n+k)) and cancel interior terms."""
    if lower.is_integer is not True or not _has_constant_numerator_quadratic_denominator(
        expression, variable
    ):
        return ()
    match = next(
        (
            (offset, shift, coefficient)
            for offset in range(-6, 7)
            for shift in range(1, 7)
            if (
                coefficient := sp.simplify(
                    expression * (variable + offset) * (variable + offset + shift)
                )
            )
            != sp.Integer(0)
            and not coefficient.has(variable)
            and sp.simplify(lower + offset).is_positive is True
        ),
        None,
    )
    if match is None:
        return ()
    offset, shift, coefficient = match
    shifted_variable = variable + offset
    decomposition = coefficient / shift * (1 / shifted_variable - 1 / (shifted_variable + shift))
    if sp.simplify(decomposition - expression) != sp.Integer(0):
        return ()
    leading_terms = sp.Add(
        *(1 / (lower + offset + index) for index in range(shift)),
    )
    match upper:
        case value if value == sp.oo:
            boundary_value = sp.simplify(coefficient * leading_terms / shift)
        case value if value.is_integer is True:
            trailing_terms = sp.Add(
                *(1 / (value + 1 + offset + index) for index in range(shift)),
            )
            boundary_value = sp.simplify(coefficient * (leading_terms - trailing_terms) / shift)
        case _:
            return ()
    if boundary_value != result:
        return ()
    decomposed_sigma = BackendSigma(
        expression=decomposition,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    return (
        BackendDerivationStep(
            rule="Decompose the summand into partial fractions",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=decomposed_sigma,
            explanation="Split the rational term into shifted reciprocals with opposite signs.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the partial fractions recovers the original summand.",
            notes=(
                BackendMathNote(
                    label="Partial-fraction identity",
                    expression=BackendIdentity(left=expression, right=decomposition),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Cancel the telescoping terms",
            before=decomposed_sigma,
            after=result,
            explanation=(
                "Write the partial sums: every interior reciprocal cancels, leaving only the "
                "boundary terms."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The surviving boundary terms simplify to the exact result.",
            notes=(
                BackendMathNote(
                    label="Surviving boundary value",
                    expression=BackendIdentity(left=boundary_value, right=result),
                ),
            ),
        ),
    )
