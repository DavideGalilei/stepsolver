"""Reusable classroom methods for structurally recognizable integrals."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendDifferential,
    BackendExpression,
    BackendIdentity,
    BackendIntegral,
    BackendIntegrationByPartsRule,
    BackendMathNote,
    BackendProduct,
    BackendSum,
)
from stepsolver.results import VerificationMethod


def _matches_result(candidate: sp.Basic, result: sp.Basic) -> bool:
    return sp.simplify(candidate - result) == sp.Integer(0)


def derive_inverse_function_by_parts(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use integration by parts for logarithms and inverse tangent."""
    if integrand not in {sp.log(variable), sp.atan(variable)}:
        return ()
    chosen_u = integrand
    chosen_v = variable
    chosen_du = sp.diff(chosen_u, variable)
    remaining_integrand = sp.simplify(chosen_v * chosen_du)
    remaining_antiderivative = sp.integrate(remaining_integrand, variable)
    formula = sp.Add(
        chosen_u * chosen_v,
        -remaining_antiderivative,
        sp.Symbol("C"),
        evaluate=False,
    )
    if remaining_antiderivative.has(sp.Integral) or not _matches_result(formula, result):
        return ()
    reduced = BackendDifference(
        left=chosen_u * chosen_v,
        right=BackendIntegral(
            integrand=remaining_integrand,
            variable=variable,
        ),
    )
    return (
        BackendDerivationStep(
            rule="Choose integration by parts",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=reduced,
            explanation=(
                "Treat the inverse function as u and 1 dx as dv, because differentiating u "
                "produces a simpler rational expression."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The integration-by-parts identity was applied exactly.",
            notes=(
                BackendMathNote(
                    label="Integration by parts",
                    expression=BackendIntegrationByPartsRule(),
                ),
                BackendMathNote(
                    label="Choose u",
                    expression=BackendIdentity(left=sp.Symbol("u"), right=chosen_u),
                ),
                BackendMathNote(
                    label="Choose dv",
                    expression=BackendIdentity(
                        left=sp.Symbol("dv"),
                        right=BackendDifferential(variable=variable),
                    ),
                ),
                BackendMathNote(
                    label="Differentiate u",
                    expression=BackendIdentity(
                        left=sp.Symbol("du"),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=chosen_du,
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Antidifferentiate dv",
                    expression=BackendIdentity(left=sp.Symbol("v"), right=chosen_v),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the simpler remaining integral",
            before=reduced,
            after=result,
            explanation="Integrate the rational remainder, collect the terms, and add C.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final expression recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Remaining antiderivative",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=remaining_integrand,
                            variable=variable,
                        ),
                        right=remaining_antiderivative,
                    ),
                ),
            ),
        ),
    )


def derive_trig_power_substitution(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Substitute sine or cosine when the matching differential is present."""
    candidates = (
        (sp.sin(variable), sp.cos(variable), sp.Integer(1), "sine"),
        (sp.cos(variable), sp.sin(variable), sp.Integer(-1), "cosine"),
    )
    for inner, differential_factor, differential_sign, name in candidates:
        power = next(
            (
                exponent
                for exponent in range(1, 9)
                if sp.simplify(
                    integrand - inner**exponent * differential_factor
                )
                == sp.Integer(0)
            ),
            None,
        )
        if power is None:
            continue
        substitution = sp.Symbol("u", real=True)
        transformed_integrand = differential_sign * substitution**power
        transformed = BackendIntegral(
            integrand=transformed_integrand,
            variable=substitution,
        )
        transformed_result = (
            differential_sign * substitution ** (power + 1) / (power + 1)
            + sp.Symbol("C")
        )
        formula = transformed_result.subs(substitution, inner)
        if not _matches_result(formula, result):
            return ()
        return (
            BackendDerivationStep(
                rule=f"Substitute the {name}",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=transformed,
                explanation=(
                    f"The differential of {name}(x) appears as a factor, leaving a power of "
                    "the new variable."
                ),
                verification_method=VerificationMethod.SUBSTITUTION,
                verification_detail="The substitution replaces both the function and differential.",
                notes=(
                    BackendMathNote(
                        label="Substitution",
                        expression=sp.Eq(substitution, inner, evaluate=False),
                    ),
                    BackendMathNote(
                        label="Differential",
                        expression=BackendIdentity(
                            left=BackendDifferential(variable=substitution),
                            right=BackendDifferential(
                                variable=variable,
                                coefficient=sp.diff(inner, variable),
                            ),
                        ),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Apply the power rule and substitute back",
                before=transformed,
                after=result,
                explanation="Integrate the power of u, then replace u with the trig function.",
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the result recovers the original product.",
            ),
        )
    return ()


def derive_inverse_tangent_substitution(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Recognize a constant multiple of u'/(1 + u^2)."""
    numerator, denominator = sp.fraction(sp.cancel(integrand))
    squared_inner = sp.simplify(denominator - 1)
    inner = sp.sqrt(squared_inner)
    if sp.simplify(inner**2 - squared_inner) != sp.Integer(0):
        return ()
    inner_derivative = sp.diff(inner, variable)
    if inner_derivative == sp.Integer(0):
        return ()
    coefficient = sp.simplify(numerator / inner_derivative)
    if coefficient.has(variable):
        return ()
    formula = coefficient * sp.atan(inner) + sp.Symbol("C")
    if not _matches_result(formula, result):
        return ()
    substitution = sp.Symbol("u", real=True)
    transformed = BackendIntegral(
        integrand=coefficient / (1 + substitution**2),
        variable=substitution,
    )
    return (
        BackendDerivationStep(
            rule="Substitute the repeated inner power",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed,
            explanation=(
                "Choose u so the denominator becomes 1 + u²; its derivative accounts for "
                "the numerator up to a constant."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The transformed integrand is a constant multiple of 1/(1+u²).",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=sp.Eq(substitution, inner, evaluate=False),
                ),
                BackendMathNote(
                    label="Differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=inner_derivative,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the arctangent antiderivative",
            before=transformed,
            after=result,
            explanation="Integrate 1/(1+u²), apply the constant multiplier, and substitute back.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final expression recovers the integrand.",
        ),
    )


def derive_cyclic_exponential_trig_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use integration by parts twice for exp(ax) times sin(bx) or cos(bx)."""
    exponential = next(
        (factor for factor in integrand.as_ordered_factors() if factor.func == sp.exp),
        None,
    )
    trig = next(
        (
            factor
            for factor in integrand.as_ordered_factors()
            if factor.func in {sp.sin, sp.cos}
        ),
        None,
    )
    if exponential is None or trig is None:
        return ()
    coefficient = sp.simplify(integrand / (exponential * trig))
    exponential_rate = sp.diff(exponential.args[0], variable)
    trig_rate = sp.diff(trig.args[0], variable)
    if (
        coefficient.has(variable)
        or exponential_rate.has(variable)
        or trig_rate.has(variable)
        or exponential_rate == sp.Integer(0)
        or trig_rate == sp.Integer(0)
    ):
        return ()
    first_term = coefficient * exponential * trig / exponential_rate
    companion = sp.cos(trig.args[0]) if trig.func == sp.sin else sp.sin(trig.args[0])
    remaining_coefficient = coefficient * trig_rate / exponential_rate
    remaining = BackendProduct(
        factors=(
            remaining_coefficient,
            BackendIntegral(
                integrand=exponential * companion,
                variable=variable,
            ),
        )
    )
    reduced: BackendExpression
    if trig.func == sp.sin:
        reduced = BackendDifference(
            left=first_term,
            right=remaining,
        )
    else:
        reduced = BackendSum(
            terms=(first_term, remaining),
        )
    if not _matches_result(sp.diff(result, variable), integrand):
        return ()
    return (
        BackendDerivationStep(
            rule="Integrate by parts once",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=reduced,
            explanation=(
                "Antidifferentiate the exponential and differentiate the trig factor. The "
                "remaining integral has the companion trig function."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The first integration-by-parts transformation is exact.",
            notes=(
                BackendMathNote(
                    label="Integration by parts",
                    expression=BackendIntegrationByPartsRule(),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate by parts again and solve for the integral",
            before=reduced,
            after=result,
            explanation=(
                "A second integration by parts makes the original integral reappear. Move it "
                "to the left, divide by its coefficient, and add C."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the solved cyclic formula recovers the integrand.",
        ),
    )
