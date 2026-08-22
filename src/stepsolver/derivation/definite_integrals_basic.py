"""Proper definite-integral derivations."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendEvaluationAtBounds,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod


def derive_definite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the Fundamental Theorem to a proper elementary definite integral."""
    if lower in {sp.oo, -sp.oo} or upper in {sp.oo, -sp.oo}:
        return ()
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    upper_value = sp.simplify(antiderivative.subs(variable, upper))
    lower_value = sp.simplify(antiderivative.subs(variable, lower))
    if sp.simplify(upper_value - lower_value - result) != sp.Integer(0):
        return ()
    evaluated_at_bounds = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    endpoint_difference = BackendDifference(left=upper_value, right=lower_value)
    generic_variable = sp.Symbol("x", real=True)
    generic_lower = sp.Symbol("a", real=True)
    generic_upper = sp.Symbol("b", real=True)
    generic_function = sp.Function("f")(generic_variable)
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=evaluated_at_bounds,
            explanation=(
                "Find an antiderivative, then evaluate it at the upper bound minus the lower bound."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the chosen antiderivative recovers the integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Fundamental Theorem",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=generic_function,
                            variable=generic_variable,
                            lower=generic_lower,
                            upper=generic_upper,
                        ),
                        right=BackendDifference(
                            left=sp.Function("F")(generic_upper),
                            right=sp.Function("F")(generic_lower),
                        ),
                    ),
                ),
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
            rule="Evaluate the bounds",
            before=evaluated_at_bounds,
            after=endpoint_difference,
            explanation=(
                "Substitute the upper and lower bounds into the antiderivative, keeping "
                "upper minus lower."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Both endpoint values were evaluated exactly.",
            notes=(
                BackendMathNote(
                    label="Upper bound",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtBounds(
                            expression=antiderivative,
                            variable=variable,
                            lower=upper,
                            upper=upper,
                        ),
                        right=upper_value,
                    ),
                ),
                BackendMathNote(
                    label="Lower bound",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtBounds(
                            expression=antiderivative,
                            variable=variable,
                            lower=lower,
                            upper=lower,
                        ),
                        right=lower_value,
                    ),
                ),
            ),
        ),
    ]
    if str(endpoint_difference) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Finish the arithmetic",
                before=endpoint_difference,
                after=result,
                explanation="Subtract the lower-bound value from the upper-bound value.",
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="The final subtraction was evaluated exactly.",
            )
        )
    return tuple(steps)
