"""Rational antiderivatives by partial fractions."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_MINIMUM_DECOMPOSED_TERMS = 2


def derive_partial_fraction_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Integrate rational functions whose partial fractions avoid logarithmic domains."""
    if not integrand.is_rational_function(variable) or result.has(sp.log):
        return ()
    decomposition = sp.apart(integrand, variable)
    terms = tuple(decomposition.as_ordered_terms())
    if str(decomposition) == str(integrand) or len(terms) < _MINIMUM_DECOMPOSED_TERMS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    if any(item.has(sp.Integral) for item in antiderivatives):
        return ()
    return (
        BackendDerivationStep(
            rule="Decompose into partial fractions",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=decomposition, variable=variable),
            explanation=(
                "Rewrite the rational function as a sum of simpler fractions that can be "
                "integrated separately."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the partial fractions recovers the original fraction.",
            notes=(
                BackendMathNote(
                    label="Partial-fraction identity",
                    expression=BackendIdentity(left=integrand, right=decomposition),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate each partial fraction",
            before=BackendIntegral(integrand=decomposition, variable=variable),
            after=result,
            explanation="Integrate the separated power and arctangent terms, then combine them.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final result recovers the rational function.",
            notes=tuple(
                BackendMathNote(
                    label=f"Term {index}",
                    expression=BackendIdentity(
                        left=BackendIntegral(integrand=term, variable=variable),
                        right=antiderivative,
                    ),
                )
                for index, (term, antiderivative) in enumerate(
                    zip(terms, antiderivatives, strict=True),
                    start=1,
                )
            ),
        ),
    )
