"""Proper definite-integral derivations."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendEvaluationAtBounds,
    BackendIdentity,
    BackendInlineMath,
    BackendIntegral,
    BackendMathNote,
    BackendSum,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_integrals import match_sine_square_root_integral


def _derive_sine_square_root_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Reduce a principal trigonometric square root to signed sine pieces."""
    matched = match_sine_square_root_integral(integrand, variable, lower, upper)
    if matched is None or sp.simplify(matched.value - result) != sp.Integer(0):
        return ()
    original = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    sine = sp.sin(variable)
    absolute_sine = sp.Abs(sine)
    square_root = sp.Pow(sine**2, sp.Rational(1, 2), evaluate=False)
    root_integral = BackendIntegral(
        integrand=square_root,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    absolute_integral = BackendIntegral(
        integrand=absolute_sine,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    steps: list[BackendDerivationStep] = []
    if integrand != absolute_sine:
        pythagorean_identity = BackendIdentity(
            left=sp.sin(variable) ** 2 + sp.cos(variable) ** 2,
            right=sp.Integer(1),
        )
        steps.append(
            BackendDerivationStep(
                rule="Use the Pythagorean identity",
                before=original,
                after=root_integral,
                explanation="Rewrite the radicand as a square of sine.",
                explanation_parts=(
                    "Use ",
                    BackendInlineMath(expression=pythagorean_identity),
                    " to rewrite the radicand.",
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The Pythagorean identity makes the two radicands equal.",
                notes=(
                    BackendMathNote(label="Pythagorean identity", expression=pythagorean_identity),
                ),
            )
        )
        generic_real = sp.Symbol("y", real=True)
        principal_root_identity = BackendIdentity(
            left=sp.Pow(generic_real**2, sp.Rational(1, 2), evaluate=False),
            right=sp.Abs(generic_real),
        )
        steps.append(
            BackendDerivationStep(
                rule="Use the principal square root",
                before=root_integral,
                after=absolute_integral,
                explanation="The principal square root of a real square is an absolute value.",
                explanation_parts=(
                    "For real ",
                    BackendInlineMath(expression=generic_real),
                    ", use ",
                    BackendInlineMath(expression=principal_root_identity),
                    ".",
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Absolute value is the nonnegative square root required by the radical."
                ),
                notes=(
                    BackendMathNote(
                        label="Principal-root rule",
                        expression=principal_root_identity,
                    ),
                ),
            )
        )

    absolute_pieces = tuple(
        BackendIntegral(
            integrand=absolute_sine,
            variable=variable,
            lower=piece_lower,
            upper=piece_upper,
        )
        for piece_lower, piece_upper in zip(
            matched.points[:-1],
            matched.points[1:],
            strict=True,
        )
    )
    signed_pieces = tuple(
        BackendIntegral(
            integrand=signed_integrand,
            variable=variable,
            lower=piece_lower,
            upper=piece_upper,
        )
        for signed_integrand, piece_lower, piece_upper in zip(
            matched.signed_integrands,
            matched.points[:-1],
            matched.points[1:],
            strict=True,
        )
    )
    absolute_piece_expression = (
        BackendSum(terms=absolute_pieces) if len(absolute_pieces) > 1 else absolute_integral
    )
    signed_piece_expression = (
        BackendSum(terms=signed_pieces)
        if len(signed_pieces) > 1
        else next(iter(signed_pieces))
    )
    if len(absolute_pieces) > 1:
        steps.append(
            BackendDerivationStep(
                rule="Split at the zeros of sine",
                before=absolute_integral,
                after=absolute_piece_expression,
                explanation="Split the interval wherever sine can change sign.",
                explanation_parts=(
                    "Split at each interior point where ",
                    BackendInlineMath(
                        expression=BackendIdentity(left=sine, right=sp.Integer(0))
                    ),
                    ". Between consecutive zeros, sine keeps one sign.",
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Additivity of definite integrals preserves the original interval."
                ),
            )
        )
    sign_notes = tuple(
        BackendMathNote(
            label=f"Sign on interval {index}",
            expression=BackendIdentity(left=absolute_piece, right=signed_piece),
        )
        for index, (absolute_piece, signed_piece) in enumerate(
            zip(absolute_pieces, signed_pieces, strict=True),
            start=1,
        )
    )
    steps.append(
        BackendDerivationStep(
            rule="Remove the absolute value on each interval",
            before=absolute_piece_expression,
            after=signed_piece_expression,
            explanation="Use the known sign of sine between each pair of consecutive zeros.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Each replacement follows the definition of absolute value.",
            notes=sign_notes,
        )
    )
    evaluated_expression = (
        BackendSum(terms=matched.piece_values)
        if len(matched.piece_values) > 1
        else next(iter(matched.piece_values))
    )
    antiderivative_notes = tuple(
        BackendMathNote(
            label=f"Antiderivative on interval {index}",
            expression=BackendIdentity(
                left=BackendDerivative(
                    expression=(
                        -sp.cos(variable) if signed_integrand == sine else sp.cos(variable)
                    ),
                    variable=variable,
                ),
                right=signed_integrand,
            ),
        )
        for index, signed_integrand in enumerate(matched.signed_integrands, start=1)
    )
    steps.append(
        BackendDerivationStep(
            rule="Evaluate each definite integral",
            before=signed_piece_expression,
            after=evaluated_expression,
            explanation="Use the appropriate sine antiderivative and evaluate both endpoints.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating each chosen antiderivative recovers its integrand."
            ),
            notes=antiderivative_notes,
        )
    )
    if len(matched.piece_values) > 1:
        steps.append(
            BackendDerivationStep(
                rule="Add the interval contributions",
                before=evaluated_expression,
                after=result,
                explanation="Add the exact area contributed by each constant-sign interval.",
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="The exact interval values sum to the stated result.",
            )
        )
    return tuple(steps)


def derive_definite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the Fundamental Theorem to a proper elementary definite integral."""
    sine_square_root_steps = _derive_sine_square_root_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if sine_square_root_steps:
        return sine_square_root_steps
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
