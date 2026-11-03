"""Improper-integral derivations and endpoint analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendEvaluationAtBounds,
    BackendIdentity,
    BackendIntegral,
    BackendLimit,
    BackendMathNote,
    BackendProduct,
    BackendSum,
    BackendUndefined,
)
from stepsolver.results import VerificationMethod

if TYPE_CHECKING:
    from collections.abc import Iterable

_UNEVALUATED_LIMIT = cast("sp.Basic", getattr(sp, "Limit"))


def derive_improper_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Evaluate a one-ended improper integral through its defining limit."""
    absolute_value_steps = _derive_absolute_value_improper_integral(
        integrand,
        variable,
        lower,
        upper,
    )
    if absolute_value_steps:
        return absolute_value_steps
    two_sided_steps = _derive_two_sided_infinite_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if two_sided_steps:
        return two_sided_steps
    singularity_steps = _derive_endpoint_singularity_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if singularity_steps:
        return singularity_steps
    interior_steps = _derive_interior_singularity_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if interior_steps:
        return interior_steps
    has_infinite_lower = lower == -sp.oo and upper not in {sp.oo, -sp.oo}
    has_infinite_upper = upper == sp.oo and lower not in {sp.oo, -sp.oo}
    if not has_infinite_lower and not has_infinite_upper:
        return ()
    bound = sp.Symbol("a" if has_infinite_lower else "b", real=True)
    finite_lower = bound if has_infinite_lower else lower
    finite_upper = upper if has_infinite_lower else bound
    approach_point = -sp.oo if has_infinite_lower else sp.oo
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    limit_of_integral = BackendLimit(
        expression=BackendIntegral(
            integrand=integrand,
            variable=variable,
            lower=finite_lower,
            upper=finite_upper,
        ),
        variable=bound,
        point=approach_point,
    )
    evaluated_at_bounds = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=finite_lower,
        upper=finite_upper,
    )
    limit_of_antiderivative = BackendLimit(
        expression=evaluated_at_bounds,
        variable=bound,
        point=approach_point,
    )
    return (
        BackendDerivationStep(
            rule="Rewrite the improper integral as a limit",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=limit_of_integral,
            explanation=(
                "Replace the infinite endpoint with a finite bound, then let that bound "
                "approach infinity."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="This limit is the definition of the improper integral.",
        ),
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=limit_of_integral,
            after=limit_of_antiderivative,
            explanation=(
                "Evaluate the finite integral with an antiderivative before taking the limit."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the antiderivative recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Chosen antiderivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=antiderivative,
                            variable=variable,
                        ),
                        right=integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the limit",
            before=limit_of_antiderivative,
            after=result,
            explanation=(
                "Evaluate the finite endpoint and take the limit at the infinite endpoint."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The endpoint limit determines whether the improper integral converges."
            ),
        ),
    )


def _derive_two_sided_infinite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Evaluate a symmetric integral over the whole real line."""
    if lower != -sp.oo or upper != sp.oo:
        return ()
    reflected = sp.simplify(integrand.subs(variable, -variable))
    is_even = sp.simplify(reflected - integrand) == sp.Integer(0)
    is_odd = sp.simplify(reflected + integrand) == sp.Integer(0)
    if not is_even and not is_odd:
        return ()
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    bound = sp.Symbol("b", real=True)
    original = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    left_tail = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=-sp.oo,
        upper=sp.Integer(0),
    )
    right_tail = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=sp.Integer(0),
        upper=sp.oo,
    )
    finite_right = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=sp.Integer(0),
        upper=bound,
    )
    tail_integral = sp.integrate(integrand, (variable, sp.Integer(0), bound))
    if tail_integral.has(sp.Integral):
        return ()
    tail_limit = sp.limit(tail_integral, bound, sp.oo)
    if tail_limit.has(_UNEVALUATED_LIMIT):
        return ()
    if is_even:
        expected = sp.simplify(2 * tail_limit)
        if not _matches_integral_value(expected, result):
            return ()
        doubled_tail = BackendProduct(factors=(sp.Integer(2), right_tail))
        limit_of_integral = BackendLimit(
            expression=finite_right,
            variable=bound,
            point=sp.oo,
        )
        doubled_limit = BackendProduct(factors=(sp.Integer(2), limit_of_integral))
        evaluated = BackendEvaluationAtBounds(
            expression=antiderivative,
            variable=variable,
            lower=sp.Integer(0),
            upper=bound,
        )
        limit_of_evaluation = BackendLimit(
            expression=evaluated,
            variable=bound,
            point=sp.oo,
        )
        doubled_evaluation = BackendProduct(factors=(sp.Integer(2), limit_of_evaluation))
        return (
            BackendDerivationStep(
                rule="Split the integral at zero",
                before=original,
                after=BackendSum(terms=(left_tail, right_tail)),
                explanation=(
                    "Split the real line at zero so each half-line can be handled as a "
                    "one-sided improper integral."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "The two half-line integrals add up to the whole-line integral."
                ),
            ),
            BackendDerivationStep(
                rule="Use symmetry of an even integrand",
                before=BackendSum(terms=(left_tail, right_tail)),
                after=doubled_tail,
                explanation=(
                    "The integrand is even, so the left tail equals the right tail. The whole "
                    "integral is twice the integral from zero to infinity."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="Substituting -x for x leaves an even integrand unchanged.",
                notes=(
                    BackendMathNote(
                        label="Even integrand",
                        expression=BackendIdentity(
                            left=integrand.subs(variable, -variable),
                            right=integrand,
                        ),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Rewrite the improper integral as a limit",
                before=doubled_tail,
                after=doubled_limit,
                explanation=(
                    "Replace the infinite endpoint with a finite bound, then let that bound "
                    "approach infinity."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "This limit is the definition of the one-sided improper integral."
                ),
            ),
            BackendDerivationStep(
                rule="Apply the Fundamental Theorem of Calculus",
                before=doubled_limit,
                after=doubled_evaluation,
                explanation=(
                    "Evaluate the finite integral with an antiderivative before taking the limit."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the antiderivative recovers the integrand.",
                notes=(
                    BackendMathNote(
                        label="Chosen antiderivative",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=integrand,
                        ),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Evaluate the limit",
                before=doubled_evaluation,
                after=result,
                explanation=(
                    "Evaluate the finite endpoint and take the limit at infinity. Doubling the "
                    "one-sided value gives the whole-line integral."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail=(
                    "The endpoint limit determines whether the improper integral converges."
                ),
            ),
        )
    if getattr(tail_limit, "is_finite", None) is not True or not _matches_integral_value(
        sp.Integer(0), result
    ):
        return ()
    evaluated = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=sp.Integer(0),
        upper=bound,
    )
    limit_of_integral = BackendLimit(
        expression=finite_right,
        variable=bound,
        point=sp.oo,
    )
    limit_of_evaluation = BackendLimit(
        expression=evaluated,
        variable=bound,
        point=sp.oo,
    )
    return (
        BackendDerivationStep(
            rule="Split the integral at zero",
            before=original,
            after=BackendSum(terms=(left_tail, right_tail)),
            explanation=(
                "Split the real line at zero so each half-line can be handled as a "
                "one-sided improper integral."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The two half-line integrals add up to the whole-line integral.",
        ),
        BackendDerivationStep(
            rule="Use symmetry of an odd integrand",
            before=BackendSum(terms=(left_tail, right_tail)),
            after=BackendDifference(left=right_tail, right=right_tail),
            explanation=(
                "The integrand is odd, so the left tail is the negative of the right tail. "
                "The two tails therefore cancel once each one converges."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Substituting -x for x changes the sign of an odd integrand.",
            notes=(
                BackendMathNote(
                    label="Odd integrand",
                    expression=BackendIdentity(
                        left=integrand.subs(variable, -variable),
                        right=-integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate one tail as a limit",
            before=limit_of_integral,
            after=tail_limit,
            explanation=(
                "Replace the infinite endpoint with a finite bound, apply the Fundamental "
                "Theorem of Calculus, and take the limit."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The one-sided tail converges, so the symmetric cancellation is valid."
            ),
            notes=(
                BackendMathNote(
                    label="Finite-bound evaluation",
                    expression=BackendIdentity(
                        left=limit_of_evaluation,
                        right=tail_integral,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Cancel the symmetric tails",
            before=BackendDifference(left=tail_limit, right=tail_limit),
            after=result,
            explanation=(
                "The two tails are equal in magnitude and opposite in sign, so their sum is zero."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Equal and opposite finite tails add to exactly zero.",
        ),
    )


def _matches_integral_value(candidate: sp.Basic, result: sp.Basic) -> bool:
    """Compare a computed improper-integral value, including divergent outcomes."""
    if result in {sp.oo, -sp.oo, sp.nan, sp.zoo}:
        return candidate == result
    return sp.simplify(candidate - result) == sp.Integer(0)


def _derive_endpoint_singularity_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Evaluate an improper integral that is unbounded at one finite endpoint."""
    if lower in {sp.oo, -sp.oo} or upper in {sp.oo, -sp.oo}:
        return ()
    try:
        singularities = cast(
            "sp.Basic",
            getattr(sp, "singularities")(integrand, variable),
        )
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if not isinstance(singularities, sp.FiniteSet):
        return ()
    try:
        interval = cast("sp.Basic", getattr(sp, "Interval")(lower, upper))
        contains = getattr(interval, "contains")
        points = [
            point
            for point in cast("Iterable[object]", singularities)
            if isinstance(point, sp.Basic) and point.is_real is True and bool(contains(point))
        ]
    except (TypeError, ValueError):
        return ()
    if len(points) != 1:
        return ()
    singular = points[0]
    if singular not in {lower, upper}:
        return ()
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    bound = sp.Symbol("t", real=True)
    finite_lower: sp.Basic
    finite_upper: sp.Basic
    if singular == lower:
        finite_lower = bound
        finite_upper = upper
        approach_point = lower
        direction = "+"
        description = "the lower endpoint"
    else:
        finite_lower = lower
        finite_upper = bound
        approach_point = upper
        direction = "-"
        description = "the upper endpoint"
    limit_expression = sp.simplify(
        antiderivative.subs(variable, finite_upper) - antiderivative.subs(variable, finite_lower)
    )
    limit_value = cast(
        "sp.Basic",
        getattr(sp, "limit")(limit_expression, bound, approach_point, dir=direction),
    )
    if limit_value.has(_UNEVALUATED_LIMIT) or not _matches_integral_value(limit_value, result):
        return ()
    original = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    limit_of_integral = BackendLimit(
        expression=BackendIntegral(
            integrand=integrand,
            variable=variable,
            lower=finite_lower,
            upper=finite_upper,
        ),
        variable=bound,
        point=approach_point,
        direction=direction,
    )
    limit_of_evaluation = BackendLimit(
        expression=BackendEvaluationAtBounds(
            expression=antiderivative,
            variable=variable,
            lower=finite_lower,
            upper=finite_upper,
        ),
        variable=bound,
        point=approach_point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Rewrite the improper integral as a limit",
            before=original,
            after=limit_of_integral,
            explanation=(
                f"The integrand is unbounded at {description}, so replace that endpoint with a "
                "finite bound and approach it from inside the interval."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="This one-sided limit is the definition of the improper integral.",
        ),
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=limit_of_integral,
            after=limit_of_evaluation,
            explanation=(
                "Evaluate the finite integral with an antiderivative before taking the limit."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the antiderivative recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Chosen antiderivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=antiderivative,
                            variable=variable,
                        ),
                        right=integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the limit",
            before=limit_of_evaluation,
            after=result,
            explanation=(
                "Take the one-sided limit at the singular endpoint and subtract the finite "
                "endpoint value."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The endpoint limit determines whether the improper integral converges."
            ),
        ),
    )


def _singularity_points(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> tuple[sp.Basic, ...] | None:
    """Return the real singularities inside a finite integration interval."""
    try:
        singularities = cast(
            "sp.Basic",
            getattr(sp, "singularities")(integrand, variable),
        )
    except (NotImplementedError, TypeError, ValueError):
        return None
    if not isinstance(singularities, sp.FiniteSet):
        return None
    try:
        interval = cast("sp.Basic", getattr(sp, "Interval")(lower, upper))
        contains = getattr(interval, "contains")
        points = tuple(
            point
            for point in cast("Iterable[object]", singularities)
            if isinstance(point, sp.Basic) and point.is_real is True and bool(contains(point))
        )
    except (TypeError, ValueError):
        return None
    return points


def _derive_interior_singularity_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Split an improper integral at a singularity inside the interval."""
    if lower in {sp.oo, -sp.oo} or upper in {sp.oo, -sp.oo}:
        return ()
    points = _singularity_points(integrand, variable, lower, upper)
    if points is None:
        return ()
    interior = tuple(point for point in points if point not in {lower, upper})
    if len(interior) != 1:
        return ()
    singular = interior[0]
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    left_bound = sp.Symbol("a", real=True)
    right_bound = sp.Symbol("b", real=True)
    left_difference = sp.simplify(
        antiderivative.subs(variable, left_bound) - antiderivative.subs(variable, lower)
    )
    right_difference = sp.simplify(
        antiderivative.subs(variable, upper) - antiderivative.subs(variable, right_bound)
    )
    try:
        left_value = cast(
            "sp.Basic",
            getattr(sp, "limit")(left_difference, left_bound, singular, dir="-"),
        )
        right_value = cast(
            "sp.Basic",
            getattr(sp, "limit")(right_difference, right_bound, singular, dir="+"),
        )
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if left_value.has(_UNEVALUATED_LIMIT) or right_value.has(_UNEVALUATED_LIMIT):
        return ()
    total = sp.simplify(left_value + right_value)
    if not _matches_integral_value(total, result):
        return ()
    both_finite = (
        getattr(left_value, "is_finite", None) is True
        and getattr(right_value, "is_finite", None) is True
    )
    original = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    left_integral = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=left_bound,
    )
    right_integral = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=right_bound,
        upper=upper,
    )
    left_limit = BackendLimit(
        expression=left_integral,
        variable=left_bound,
        point=singular,
        direction="-",
    )
    right_limit = BackendLimit(
        expression=right_integral,
        variable=right_bound,
        point=singular,
        direction="+",
    )
    if both_finite:
        conclusion = BackendDerivationStep(
            rule="Add the two one-sided values",
            before=BackendSum(terms=(left_value, right_value)),
            after=total,
            explanation=(
                "Both pieces converge, so the improper integral is their sum. A convergent "
                "interior singularity must be approached independently from each side."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "The two one-sided improper integrals are finite and add to the displayed value."
            ),
        )
    else:
        conclusion = BackendDerivationStep(
            rule="Conclude the integral diverges",
            before=BackendSum(terms=(left_value, right_value)),
            after=BackendUndefined() if total in {sp.nan, sp.zoo} else total,
            explanation=(
                "At least one piece does not converge. The two one-sided limits are taken "
                "independently, so an infinite value on one side cannot be cancelled by the "
                "other side."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail=(
                "A convergent improper integral requires both one-sided pieces to be finite."
            ),
        )
    return (
        BackendDerivationStep(
            rule="Split the integral at the interior singularity",
            before=original,
            after=BackendSum(terms=(left_integral, right_integral)),
            explanation=(
                "The integrand is unbounded inside the interval. Split the integral at that "
                "point and evaluate each piece as a separate one-sided improper integral."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail=(
                "The split is valid because both endpoints of the whole interval are finite."
            ),
            notes=(
                BackendMathNote(
                    label="Interior singularity",
                    expression=BackendIdentity(
                        left=variable,
                        right=singular,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the left piece as a limit",
            before=left_limit,
            after=left_value,
            explanation=(
                "Replace the singular upper endpoint with a finite bound and approach the "
                "singularity from the left."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The one-sided limit gives the exact left-piece value.",
        ),
        BackendDerivationStep(
            rule="Evaluate the right piece as a limit",
            before=right_limit,
            after=right_value,
            explanation=(
                "Replace the singular lower endpoint with a finite bound and approach the "
                "singularity from the right."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The one-sided limit gives the exact right-piece value.",
        ),
        conclusion,
    )


def _derive_absolute_value_improper_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Show why the standard improper integrals of ``abs(x)`` diverge."""
    if integrand != sp.Abs(variable):
        return ()
    original = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if lower == -sp.oo and upper == sp.oo:
        left_integral = BackendIntegral(
            integrand=integrand,
            variable=variable,
            lower=-sp.oo,
            upper=sp.Integer(0),
        )
        right_integral = BackendIntegral(
            integrand=integrand,
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        )
        signed_integrals = BackendSum(
            terms=(
                BackendIntegral(
                    integrand=-variable,
                    variable=variable,
                    lower=-sp.oo,
                    upper=sp.Integer(0),
                ),
                BackendIntegral(
                    integrand=variable,
                    variable=variable,
                    lower=sp.Integer(0),
                    upper=sp.oo,
                ),
            )
        )
        a = sp.Symbol("a", real=True)
        b = sp.Symbol("b", real=True)
        tail_limits = BackendSum(
            terms=(
                BackendLimit(
                    expression=a**2 / 2,
                    variable=a,
                    point=-sp.oo,
                ),
                BackendLimit(
                    expression=b**2 / 2,
                    variable=b,
                    point=sp.oo,
                ),
            )
        )
        return (
            BackendDerivationStep(
                rule="Split the integral at zero",
                before=original,
                after=BackendSum(terms=(left_integral, right_integral)),
                explanation=(
                    "The formula for absolute value changes at zero, so treat the two "
                    "half-lines separately."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The real line is split at the only sign-change point.",
            ),
            BackendDerivationStep(
                rule="Remove the absolute value on each interval",
                before=BackendSum(terms=(left_integral, right_integral)),
                after=signed_integrals,
                explanation=("For x at or below zero, |x| = -x; for x at or above zero, |x| = x."),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="Each replacement uses the sign of x on its interval.",
                notes=(
                    BackendMathNote(
                        label="Left half-line",
                        expression=BackendIdentity(left=sp.Abs(variable), right=-variable),
                    ),
                    BackendMathNote(
                        label="Right half-line",
                        expression=BackendIdentity(left=sp.Abs(variable), right=variable),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Evaluate both improper tails",
                before=signed_integrals,
                after=tail_limits,
                explanation=(
                    "Use an antiderivative on each finite interval, then send its outer "
                    "endpoint toward infinity."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "Differentiating -x^2/2 and x^2/2 gives -x and x respectively."
                ),
            ),
            BackendDerivationStep(
                rule="Check convergence of each tail",
                before=tail_limits,
                after=sp.oo,
                explanation=(
                    "Both limits grow without bound. Since even one divergent tail is "
                    "enough, the integral over the whole real line diverges."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="Both quadratic endpoint limits equal +infinity.",
            ),
        )

    has_infinite_lower = lower == -sp.oo and upper == sp.Integer(0)
    has_infinite_upper = lower == sp.Integer(0) and upper == sp.oo
    if not has_infinite_lower and not has_infinite_upper:
        return ()
    transformed_integrand = -variable if has_infinite_lower else variable
    bound = sp.Symbol("a" if has_infinite_lower else "b", real=True)
    finite_lower = bound if has_infinite_lower else lower
    finite_upper = upper if has_infinite_lower else bound
    approach_point = -sp.oo if has_infinite_lower else sp.oo
    transformed = BackendIntegral(
        integrand=transformed_integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    finite_integral = BackendIntegral(
        integrand=transformed_integrand,
        variable=variable,
        lower=finite_lower,
        upper=finite_upper,
    )
    limit_of_integral = BackendLimit(
        expression=finite_integral,
        variable=bound,
        point=approach_point,
    )
    antiderivative = sp.integrate(transformed_integrand, variable)
    evaluated = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=finite_lower,
        upper=finite_upper,
    )
    endpoint_value = sp.integrate(
        transformed_integrand,
        (variable, finite_lower, finite_upper),
    )
    endpoint_limit = BackendLimit(
        expression=endpoint_value,
        variable=bound,
        point=approach_point,
    )
    interval_description = "x <= 0" if has_infinite_lower else "x >= 0"
    replacement = "-x" if has_infinite_lower else "x"
    return (
        BackendDerivationStep(
            rule="Use the sign of x on the interval",
            before=original,
            after=transformed,
            explanation=(
                f"Throughout this interval, {interval_description}, so |x| = {replacement}."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The absolute-value definition was applied on the interval.",
            notes=(
                BackendMathNote(
                    label="Absolute value on this interval",
                    expression=BackendIdentity(
                        left=sp.Abs(variable),
                        right=transformed_integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Rewrite the improper integral as a limit",
            before=transformed,
            after=limit_of_integral,
            explanation=(
                "Replace the infinite endpoint with a finite bound, then move that bound "
                "toward infinity."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="This limit is the definition of the improper integral.",
        ),
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=limit_of_integral,
            after=BackendLimit(
                expression=evaluated,
                variable=bound,
                point=approach_point,
            ),
            explanation="Evaluate the finite integral with an antiderivative first.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the antiderivative recovers the integrand.",
        ),
        BackendDerivationStep(
            rule="Evaluate the finite bounds",
            before=BackendLimit(
                expression=evaluated,
                variable=bound,
                point=approach_point,
            ),
            after=endpoint_limit,
            explanation="Substitute the two finite endpoints and simplify.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The endpoint subtraction simplifies exactly.",
        ),
        BackendDerivationStep(
            rule="Test the endpoint limit",
            before=endpoint_limit,
            after=sp.oo,
            explanation=(
                "The quadratic term grows without bound, so the improper integral "
                "diverges to +infinity."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The quadratic endpoint limit equals +infinity.",
        ),
    )
