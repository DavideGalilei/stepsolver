"""Conditional convergence of alternating p-series."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendSigma,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_series import match_alternating_p_series


def derive_alternating_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the alternating-series criterion."""
    match = match_alternating_p_series(expression, variable, lower, upper)
    if match is None or match.value != result:
        return ()
    partial_upper = sp.Symbol("N", integer=True, positive=True)
    starting_coefficient = -match.coefficient
    canonical_expression = (
        starting_coefficient * (-sp.Integer(1)) ** (variable + 1) / variable**match.power
    )
    partial_sum_limit = BackendLimit(
        expression=BackendSigma(
            expression=canonical_expression,
            variable=variable,
            lower=sp.Integer(1),
            upper=partial_upper,
        ),
        variable=partial_upper,
        point=sp.oo,
    )
    magnitude = variable ** (-match.power)
    first_step = BackendDerivationStep(
        rule="Apply the Leibniz alternating-series test",
        before=BackendSigma(
            expression=expression,
            variable=variable,
            lower=lower,
            upper=upper,
        ),
        after=partial_sum_limit,
        explanation=(
            "Factor out the fixed overall coefficient. The remaining term signs alternate, "
            "and their positive magnitudes decrease to zero. Therefore the partial sums "
            "converge by the Leibniz test."
        ),
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail=(
            "The positivity, monotone decrease, and zero-limit conditions were checked."
        ),
        notes=(
            BackendMathNote(
                label="Core magnitude limit",
                expression=BackendIdentity(
                    left=BackendLimit(
                        expression=magnitude,
                        variable=variable,
                        point=sp.oo,
                    ),
                    right=sp.Integer(0),
                ),
            ),
            BackendMathNote(
                label="Decreasing core magnitudes",
                expression=sp.Lt(
                    (variable + 1) ** (-match.power),
                    magnitude,
                ),
            ),
        ),
    )
    generic_power = sp.Symbol("p", real=True, positive=True)
    if match.power == 1:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=(-sp.Integer(1)) ** (variable + 1) / variable,
                variable=variable,
                lower=sp.Integer(1),
                upper=sp.oo,
            ),
            right=sp.log(sp.Integer(2)),
        )
        rule = "Use the alternating harmonic-series identity"
        explanation = "The convergent alternating harmonic series has exact value log(2)."
    else:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=(-sp.Integer(1)) ** (variable + 1) / variable**generic_power,
                variable=variable,
                lower=sp.Integer(1),
                upper=sp.oo,
            ),
            right=(1 - sp.Integer(2) ** (1 - generic_power)) * sp.zeta(generic_power),
        )
        rule = "Use the Dirichlet eta identity"
        explanation = (
            "Relate the convergent alternating p-series to the Riemann zeta function, "
            f"then substitute p = {match.power} and the overall coefficient."
        )
    return (
        first_step,
        BackendDerivationStep(
            rule=rule,
            before=partial_sum_limit,
            after=result,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The standard alternating-series identity gives the exact value.",
            notes=(BackendMathNote(label="Exact identity", expression=identity),),
        ),
    )
