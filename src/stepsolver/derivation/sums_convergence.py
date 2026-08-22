"""Convergence tests and special infinite-series strategies."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendExpression,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNthRoot,
    BackendProduct,
    BackendQuotient,
    BackendSigma,
    BackendSum,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_series import match_alternating_p_series, match_harmonic_sine_series
from stepsolver.sympy_support import contains_unevaluated_operation

_GEOMETRIC_POWER_ARITY = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class NthRootPowerMatch:
    """A positive base raised to and rooted by the summation index."""

    root_term: sp.Basic
    radicand: sp.Basic
    base: sp.Basic
    coefficient: sp.Basic


def _scaled_expression(
    amplitude: sp.Basic,
    expression: BackendExpression,
) -> BackendExpression:
    if amplitude == sp.Integer(1):
        return expression
    return BackendProduct(factors=(amplitude, expression))


def _derive_harmonic_sine_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
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


def _derive_alternating_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
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


def _derive_nth_term_divergence(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
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


def _derive_nth_root_power_divergence(
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


CONVERGENCE_SUM_STRATEGIES = (
    _derive_harmonic_sine_series,
    _derive_alternating_p_series,
    _derive_nth_root_power_divergence,
    _derive_nth_term_divergence,
)
