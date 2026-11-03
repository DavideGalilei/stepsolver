"""Logarithmic antiderivatives for constant-over-linear integrands."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
    BackendNotEqual,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_equivalence import expressions_equivalent
from stepsolver.sympy_integrals import (
    ReciprocalLinearIntegral,
    match_reciprocal_linear_integral,
)
from stepsolver.sympy_support import (
    negative_condition,
    positive_condition,
    refine_expression,
)


def _differentiates_on_real_intervals(
    antiderivative: sp.Basic,
    integrand: sp.Basic,
    variable: sp.Symbol,
    denominator: sp.Basic,
) -> bool:
    """Verify the antiderivative away from the denominator's real zero."""
    difference = sp.diff(antiderivative, variable) - integrand
    conditions = (
        positive_condition(denominator),
        negative_condition(denominator),
    )
    return all(
        expressions_equivalent(refine_expression(difference, condition), sp.Integer(0))
        for condition in conditions
    )


def _constraints(
    matched: ReciprocalLinearIntegral,
    variable: sp.Symbol,
) -> tuple[BackendStepConstraint, ...]:
    constraints = [
        BackendStepConstraint(
            explanation="The denominator cannot equal zero.",
            expression=BackendNotEqual(
                left=matched.denominator,
                right=sp.Integer(0),
            ),
        )
    ]
    if not expressions_equivalent(matched.denominator, variable):
        constraints.append(
            BackendStepConstraint(
                explanation="This value is outside the domain of the integrand.",
                expression=BackendNotEqual(
                    left=variable,
                    right=matched.root,
                ),
            )
        )
    return tuple(constraints)


def derive_reciprocal_linear_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive ``c / (a*x + b)`` using the logarithmic rule with an absolute value."""
    matched = match_reciprocal_linear_integral(integrand, variable)
    if matched is None:
        return ()
    if not expressions_equivalent(matched.result, result):
        return ()
    if not _differentiates_on_real_intervals(
        matched.antiderivative,
        integrand,
        variable,
        matched.denominator,
    ):
        return ()
    substitution_variable = sp.Symbol("u", real=True)
    unit_integrand = sp.Pow(substitution_variable, -1, evaluate=False)
    displayed_coefficient = None if matched.coefficient == sp.Integer(1) else matched.coefficient
    integration_constant = sp.Symbol("C")
    logarithm_in_substitution_variable = sp.log(sp.Abs(substitution_variable))
    logarithm_in_original_variable = sp.log(sp.Abs(matched.denominator))
    formula_in_substitution_variable_term = (
        logarithm_in_substitution_variable
        if displayed_coefficient is None
        else sp.Mul(
            displayed_coefficient,
            logarithm_in_substitution_variable,
            evaluate=False,
        )
    )
    formula_term = (
        logarithm_in_original_variable
        if displayed_coefficient is None
        else sp.Mul(
            displayed_coefficient,
            logarithm_in_original_variable,
            evaluate=False,
        )
    )
    formula_in_substitution_variable = sp.Add(
        formula_in_substitution_variable_term,
        integration_constant,
        evaluate=False,
    )
    formula = sp.Add(formula_term, integration_constant, evaluate=False)
    transformed_integral = BackendIntegral(
        integrand=unit_integrand,
        variable=substitution_variable,
        coefficient=displayed_coefficient,
    )
    return (
        BackendDerivationStep(
            rule="Substitute the linear denominator",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "The denominator is linear, so its derivative is a constant. Use the "
                "denominator as the new variable."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the denominator and its differential recovers the original integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=sp.Eq(
                        substitution_variable,
                        matched.denominator,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Differentiate the substitution",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution_variable),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=matched.denominator_derivative,
                        ),
                    ),
                ),
            ),
            introduced_constraints=_constraints(matched, variable),
        ),
        BackendDerivationStep(
            rule="Use the logarithmic rule for nonzero inputs",
            before=transformed_integral,
            after=formula_in_substitution_variable,
            explanation=(
                "The reciprocal integrates to the natural logarithm of the absolute value. "
                "The absolute value keeps the antiderivative real on both sides of the "
                "excluded value."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=("Differentiating log(abs(u)) gives 1/u wherever u is nonzero."),
            notes=(
                BackendMathNote(
                    label="Rule for nonzero inputs",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=unit_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.Add(
                            logarithm_in_substitution_variable,
                            integration_constant,
                            evaluate=False,
                        ),
                    ),
                ),
                BackendMathNote(
                    label="The logarithm is defined away from zero",
                    expression=BackendNotEqual(
                        left=substitution_variable,
                        right=sp.Integer(0),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=formula_in_substitution_variable,
            after=formula,
            explanation=(
                "Replace the temporary variable with the original linear denominator. "
                "The result is valid on each interval where that denominator is nonzero."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Direct substitution gives an antiderivative equivalent to the exact result."
            ),
            notes=(
                BackendMathNote(
                    label="Replace the temporary variable",
                    expression=sp.Eq(
                        substitution_variable,
                        matched.denominator,
                        evaluate=False,
                    ),
                ),
            ),
        ),
    )
