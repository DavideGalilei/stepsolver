"""Derivations of sine and inverse-tangent series."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendExpression,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendProduct,
    BackendQuotient,
    BackendSigma,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_series import match_harmonic_sine_series


def _scaled_expression(
    amplitude: sp.Basic,
    expression: BackendExpression,
) -> BackendExpression:
    if amplitude == sp.Integer(1):
        return expression
    return BackendProduct(factors=(amplitude, expression))


def derive_harmonic_sine_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use the Abel limit for the sine series."""
    match = match_harmonic_sine_series(expression, variable, lower, upper)
    if match is None or match.value != result:
        return ()
    sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if match.normalized_angle == sp.Integer(0):
        return (
            BackendDerivationStep(
                rule="Simplify the sine terms",
                before=sigma,
                after=sp.Integer(0),
                explanation=(
                    "The angle is a whole number of full turns, so every sine term is zero."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="Each summand simplifies exactly to zero.",
            ),
        )

    damping = sp.Symbol("r", real=True, positive=True)
    generic_angle = sp.Symbol("theta", real=True, positive=True)
    partial_index = sp.Symbol("k", integer=True, positive=True)
    partial_upper = sp.Symbol("N", integer=True, positive=True)
    damped_sigma = BackendSigma(
        expression=(
            match.amplitude
            * damping**variable
            * sp.sin(variable * match.normalized_angle)
            / variable
        ),
        variable=variable,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    damped_limit = BackendLimit(
        expression=damped_sigma,
        variable=damping,
        point=sp.Integer(1),
        direction="-",
    )
    arctangent = _scaled_expression(
        match.amplitude,
        sp.atan(
            damping
            * sp.sin(match.normalized_angle)
            / (1 - damping * sp.cos(match.normalized_angle))
        ),
    )
    arctangent_limit = BackendLimit(
        expression=arctangent,
        variable=damping,
        point=sp.Integer(1),
        direction="-",
    )
    sawtooth_value = _scaled_expression(
        match.amplitude,
        BackendQuotient(
            numerator=BackendDifference(left=sp.pi, right=match.normalized_angle),
            denominator=sp.Integer(2),
        ),
    )
    return (
        BackendDerivationStep(
            rule="Introduce an Abel convergence factor",
            before=sigma,
            after=damped_limit,
            explanation=(
                "First reduce the angle modulo 2*pi, which does not change any sine term. "
                "The factors 1/n decrease to zero, while the partial sums of the sine terms "
                "stay bounded. Dirichlet's test proves convergence, and Abel's theorem lets "
                "us approach the series through the easier damped series with 0 < r < 1."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Dirichlet's test and Abel's limit theorem apply to the harmonic sine series."
            ),
            notes=(
                BackendMathNote(
                    label="Coefficient limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=1 / variable,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
                BackendMathNote(
                    label="Normalized angle",
                    expression=BackendIdentity(
                        left=generic_angle,
                        right=match.normalized_angle,
                    ),
                ),
                BackendMathNote(
                    label="Sine partial-sum identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=sp.sin(partial_index * generic_angle),
                            variable=partial_index,
                            lower=sp.Integer(1),
                            upper=partial_upper,
                        ),
                        right=(
                            sp.sin(partial_upper * generic_angle / 2)
                            * sp.sin((partial_upper + 1) * generic_angle / 2)
                            / sp.sin(generic_angle / 2)
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Sum the damped sine series",
            before=damped_limit,
            after=arctangent_limit,
            explanation=(
                "For 0 < r < 1, integrate the geometric series term by term and take its "
                "imaginary part. This gives an arctangent closed form."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The damped identity follows from the absolutely convergent geometric series."
            ),
            notes=(
                BackendMathNote(
                    label="Damped sine-series identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=(
                                damping**variable * sp.sin(variable * generic_angle) / variable
                            ),
                            variable=variable,
                            lower=sp.Integer(1),
                            upper=sp.oo,
                        ),
                        right=sp.atan(
                            damping * sp.sin(generic_angle) / (1 - damping * sp.cos(generic_angle))
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Remove the convergence factor",
            before=arctangent_limit,
            after=sawtooth_value,
            explanation=(
                "Let r approach 1 from below. For an angle between 0 and 2*pi, the "
                "arctangent limit is (pi - theta)/2."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The one-sided limit gives the Fourier sawtooth value.",
            notes=(
                BackendMathNote(
                    label="Harmonic sine-series identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=sp.sin(variable * generic_angle) / variable,
                            variable=variable,
                            lower=sp.Integer(1),
                            upper=sp.oo,
                        ),
                        right=BackendQuotient(
                            numerator=BackendDifference(left=sp.pi, right=generic_angle),
                            denominator=sp.Integer(2),
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Simplify the exact value",
            before=sawtooth_value,
            after=result,
            explanation="Substitute the normalized angle and simplify the exact arithmetic.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The displayed expression simplifies to the exact result.",
        ),
    )


def derive_gregory_leibniz_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use the arctangent power series at x = 1."""
    expected_term = (-1) ** variable / (2 * variable + 1)
    if (
        lower != sp.Integer(0)
        or upper != sp.oo
        or sp.simplify(expression - expected_term) != sp.Integer(0)
        or sp.simplify(result - sp.pi / 4) != sp.Integer(0)
    ):
        return ()
    generic = sp.Symbol("x", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=(-1) ** variable * generic ** (2 * variable + 1) / (2 * variable + 1),
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        ),
        right=sp.atan(generic),
    )
    return (
        BackendDerivationStep(
            rule="Apply the Gregory-Leibniz arctangent series",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation="Set x = 1 in the arctangent power series, so arctan(1) = pi/4.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The convergent endpoint value of the arctangent series applies.",
            notes=(BackendMathNote(label="Arctangent series", expression=identity),),
        ),
    )
