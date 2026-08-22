"""Parameterized and Dirichlet-type integral derivations."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendLimit,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod


def derive_dirichlet_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the classical Dirichlet integral with a damping parameter."""
    scaled_sine = sp.simplify(integrand * variable)
    if scaled_sine.func != sp.sin or len(scaled_sine.args) != 1:
        return ()
    sine_argument = scaled_sine.args[0]
    frequency = sp.simplify(sine_argument / variable)
    if frequency.has(variable) or frequency.is_positive is not True:
        return ()
    if str(sp.simplify(lower)) != "0" or upper != sp.oo:
        return ()
    if str(sp.simplify(result - sp.pi / 2)) != "0":
        return ()
    expected_integrand = sp.sin(variable) / variable
    if frequency != sp.Integer(1):
        transformed_variable = sp.Symbol("t", positive=True)
        transformed_integral = BackendIntegral(
            integrand=sp.sin(transformed_variable) / transformed_variable,
            variable=transformed_variable,
            lower=lower,
            upper=upper,
        )
        substitution_step = BackendDerivationStep(
            rule="Scale the integration variable",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=transformed_integral,
            explanation=(
                "Scale the variable so the sine argument becomes the new integration variable; "
                "the scale factors cancel."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the variable and differential produces the standard Dirichlet integral."
            ),
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=sp.Eq(
                        transformed_variable,
                        sine_argument,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Rewrite the original variable",
                    expression=sp.Eq(
                        variable,
                        transformed_variable / frequency,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=transformed_variable,
                            coefficient=1 / frequency,
                        ),
                    ),
                ),
            ),
        )
        standard_steps = derive_dirichlet_integral(
            sp.sin(transformed_variable) / transformed_variable,
            transformed_variable,
            lower,
            upper,
            result,
        )
        return (substitution_step, *standard_steps)

    parameter = sp.Symbol("a", positive=True)
    function = sp.Function("F")(parameter)
    integration_constant = sp.Symbol("C")
    damping = sp.exp(-parameter * variable)
    damped_integrand = damping * expected_integrand
    laplace_integrand = damping * sp.sin(variable)
    derivative_value = -1 / (parameter**2 + 1)
    general_function = sp.Add(
        integration_constant,
        -sp.atan(parameter),
        evaluate=False,
    )
    resolved_function = sp.Add(
        sp.pi / 2,
        -sp.atan(parameter),
        evaluate=False,
    )
    original_integral = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    damped_integral = BackendIntegral(
        integrand=damped_integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    parameter_limit = BackendLimit(
        expression=function,
        variable=parameter,
        point=sp.Integer(0),
        direction="+",
    )
    derivative_identity = BackendIdentity(
        left=BackendDerivative(expression=function, variable=parameter),
        right=derivative_value,
    )
    return (
        BackendDerivationStep(
            rule="Introduce a damping parameter",
            before=original_integral,
            after=parameter_limit,
            explanation=(
                "Temporarily multiply by an exponential damping factor. This makes the "
                "parameterized integral easier to differentiate, and the original integral "
                "is recovered as the damping disappears."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The Abel limit of the damped integral equals the original improper integral."
            ),
            notes=(
                BackendMathNote(
                    label="Define the damped integral",
                    expression=BackendIdentity(left=function, right=damped_integral),
                ),
                BackendMathNote(
                    label="Remove the damping at the end",
                    expression=BackendIdentity(left=original_integral, right=parameter_limit),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Differentiate with respect to the parameter",
            before=BackendIdentity(left=function, right=damped_integral),
            after=derivative_identity,
            explanation=(
                "Differentiating the damping factor cancels the division by the integration "
                "variable, leaving a standard Laplace integral."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating under the integral sign produces the displayed Laplace integral."
            ),
            notes=(
                BackendMathNote(
                    label="Differentiate the integrand",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=damped_integrand,
                            variable=parameter,
                        ),
                        right=-laplace_integrand,
                    ),
                ),
                BackendMathNote(
                    label="Evaluate the remaining Laplace integral",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=laplace_integrand,
                            variable=variable,
                            lower=lower,
                            upper=upper,
                        ),
                        right=1 / (parameter**2 + 1),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Recover the parameterized integral",
            before=derivative_identity,
            after=BackendIdentity(left=function, right=general_function),
            explanation="Integrate with respect to the parameter to recover the function.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the recovered expression gives the parameter derivative."
            ),
            notes=(
                BackendMathNote(
                    label="Arctangent derivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=sp.atan(parameter),
                            variable=parameter,
                        ),
                        right=1 / (parameter**2 + 1),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Determine the constant",
            before=BackendIdentity(left=function, right=general_function),
            after=BackendIdentity(left=function, right=resolved_function),
            explanation=(
                "As the damping becomes infinitely strong, the integral tends to zero. Use "
                "that boundary value to determine the integration constant."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The boundary limits force the constant to equal pi over two.",
            notes=(
                BackendMathNote(
                    label="The damped integral vanishes",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=function,
                            variable=parameter,
                            point=sp.oo,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
                BackendMathNote(
                    label="Arctangent limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=sp.atan(parameter),
                            variable=parameter,
                            point=sp.oo,
                        ),
                        right=sp.pi / 2,
                    ),
                ),
                BackendMathNote(
                    label="Therefore",
                    expression=sp.Eq(integration_constant, sp.pi / 2, evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Remove the damping",
            before=parameter_limit,
            after=result,
            explanation=(
                "Let the damping parameter approach zero from the positive side to recover "
                "the original improper integral."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The right-hand limit of the resolved expression is pi over two.",
            notes=(
                BackendMathNote(
                    label="Use the resolved formula",
                    expression=BackendIdentity(
                        left=parameter_limit,
                        right=BackendLimit(
                            expression=resolved_function,
                            variable=parameter,
                            point=sp.Integer(0),
                            direction="+",
                        ),
                    ),
                ),
            ),
        ),
    )
