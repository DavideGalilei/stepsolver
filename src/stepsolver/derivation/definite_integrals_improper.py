"""Improper-integral derivations and endpoint analysis."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendEvaluationAtBounds,
    BackendIdentity,
    BackendIntegral,
    BackendLimit,
    BackendMathNote,
    BackendSum,
)
from stepsolver.results import VerificationMethod


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
