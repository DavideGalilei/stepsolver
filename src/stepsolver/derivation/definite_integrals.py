"""Finite, improper, and parameterized definite-integral derivations."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendDifferential,
    BackendEvaluationAtBounds,
    BackendIdentity,
    BackendIntegral,
    BackendLimit,
    BackendMathNote,
    BackendSum,
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
