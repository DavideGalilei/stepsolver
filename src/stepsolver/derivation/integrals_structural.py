"""Reusable classroom methods for structurally recognizable integrals."""

from __future__ import annotations

from typing import cast

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

_POWER_ARITY = 2


def _matches_result(candidate: sp.Basic, result: sp.Basic) -> bool:
    return sp.simplify(candidate - result) == sp.Integer(0)


def _matches_antiderivative(
    candidate: sp.Basic,
    result: sp.Basic,
    variable: sp.Symbol,
) -> bool:
    """Compare antiderivatives modulo an additive constant."""
    return sp.simplify(sp.diff(candidate - result, variable)) == sp.Integer(0)


def _matching_functions(
    expression: sp.Basic,
    function: object,
) -> tuple[sp.Basic, ...]:
    """Collect applications of one function without relying on broad tree queries."""
    current = (expression,) if expression.func == function else ()
    nested = tuple(
        match
        for argument in expression.args
        for match in _matching_functions(argument, function)
    )
    return (*current, *nested)


def derive_inverse_function_by_parts(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use integration by parts for logarithms and inverse tangent of affine inputs."""
    _, inverse_function = integrand.as_independent(variable, as_Add=False)
    if (
        inverse_function.func not in {sp.log, sp.atan}
        or len(inverse_function.args) != 1
    ):
        return ()
    inner = inverse_function.args[0]
    inner_rate = sp.diff(inner, variable)
    if inner_rate == sp.Integer(0) or inner_rate.has(variable):
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
    if remaining_antiderivative.has(sp.Integral) or not _matches_antiderivative(
        formula,
        result,
        variable,
    ):
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
                "Treat the logarithm or inverse tangent as u and 1 dx as dv. Its affine "
                "inner expression has a constant derivative, leaving a simpler rational "
                "integral."
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
    sine_candidates = tuple(
        (inner, sp.cos(inner.args[0]), "sine")
        for inner in sorted(_matching_functions(integrand, sp.sin), key=str)
    )
    cosine_candidates = tuple(
        (inner, sp.sin(inner.args[0]), "cosine")
        for inner in sorted(_matching_functions(integrand, sp.cos), key=str)
    )
    for inner, differential_factor, name in (*sine_candidates, *cosine_candidates):
        derivative_coefficient = sp.simplify(
            sp.diff(inner, variable) / differential_factor
        )
        if (
            derivative_coefficient == sp.Integer(0)
            or derivative_coefficient.has(variable)
        ):
            continue
        for power in range(1, 9):
            coefficient = sp.simplify(
                integrand / (inner**power * differential_factor)
            )
            if coefficient.has(variable):
                continue
            transformed_coefficient = sp.simplify(
                coefficient / derivative_coefficient
            )
            break
        else:
            continue
        substitution = sp.Symbol("u", real=True)
        transformed_integrand = transformed_coefficient * substitution**power
        transformed = BackendIntegral(
            integrand=transformed_integrand,
            variable=substitution,
        )
        transformed_result = (
            transformed_coefficient * substitution ** (power + 1) / (power + 1)
            + sp.Symbol("C")
        )
        formula = transformed_result.subs(substitution, inner)
        if not _matches_antiderivative(formula, result, variable):
            continue
        return (
            BackendDerivationStep(
                rule=f"Substitute the {name}",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=transformed,
                explanation=(
                    f"The differential of the {name} expression appears up to a constant "
                    "factor, leaving a power of the new variable."
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


def _one_plus_square_inner(expression: sp.Basic) -> sp.Basic | None:
    """Return u from an expression structurally equal to 1 + u**2."""
    terms = expression.args if expression.func == sp.Add else (expression,)
    for term in terms:
        if (
            term.is_Pow
            and len(term.args) == _POWER_ARITY
            and term.args[1] == sp.Integer(2)
            and sp.simplify(expression - term) == sp.Integer(1)
        ):
            return term.args[0]
    return None


def _perfect_square_base(expression: sp.Basic) -> sp.Basic | None:
    """Recover an algebraic base whose square is the supplied expression."""
    factored = sp.factor(expression)
    if factored.func == sp.exp and len(factored.args) == 1:
        return sp.exp(factored.args[0] / 2)
    if factored.is_Pow and len(factored.args) == _POWER_ARITY:
        base, exponent = factored.args
        half_exponent = exponent / 2
        if exponent.is_integer is True and half_exponent.is_integer is True:
            return cast("sp.Basic", base**half_exponent)
    if factored.is_Mul:
        bases = tuple(_perfect_square_base(factor) for factor in factored.args)
        if all(base is not None for base in bases):
            return sp.Mul(*(base for base in bases if base is not None))
    if not factored.has(*expression.free_symbols):
        root = sp.sqrt(factored)
        if sp.simplify(root**2 - factored) == sp.Integer(0):
            return root
    return None


def derive_inverse_tangent_substitution(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Recognize a constant multiple of u'/(1 + u^2)."""
    numerator, denominator = sp.fraction(sp.cancel(integrand))
    if not numerator.has(variable):
        return ()
    inner = _one_plus_square_inner(denominator)
    if inner is None:
        squared_inner = sp.simplify(denominator - 1)
        inner = _perfect_square_base(squared_inner)
        if inner is None or sp.simplify(inner**2 - squared_inner) != sp.Integer(0):
            return ()
    inner_derivative = sp.diff(inner, variable)
    if inner_derivative == sp.Integer(0):
        return ()
    coefficient = sp.simplify(numerator / inner_derivative)
    if coefficient.has(variable):
        return ()
    formula = coefficient * sp.atan(inner) + sp.Symbol("C")
    if not _matches_antiderivative(formula, result, variable):
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
