"""Completed-square derivation for reciprocal quadratic integrals."""

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifferential,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReciprocalQuadraticDerivation:
    """Values shared by the display steps for one completed-square derivation."""

    integrand: sp.Basic
    variable: sp.Symbol
    denominator: sp.Basic
    coefficient_a: sp.Basic
    coefficient_b: sp.Basic
    coefficient_c: sp.Basic
    radius_squared: sp.Basic
    completed_denominator: sp.Basic
    completed_integrand: sp.Basic
    shift: sp.Basic
    radius: sp.Basic
    substitution_variable: sp.Symbol
    unit_integrand: sp.Basic
    normalized_argument: sp.Basic
    normalized_coefficient: sp.Basic
    coefficient_is_one: bool
    formula_in_substitution_variable: sp.Basic
    formula: sp.Basic
    integration_constant: sp.Symbol


def _completion_notes(
    *,
    denominator: sp.Basic,
    variable: sp.Symbol,
    coefficient_a: sp.Basic,
    coefficient_b: sp.Basic,
    coefficient_c: sp.Basic,
    radius_squared: sp.Basic,
    completed_denominator: sp.Basic,
) -> tuple[BackendMathNote, ...]:
    """Build a concrete, student-first completing-the-square explanation."""
    normalized_linear_coefficient = sp.simplify(coefficient_b / coefficient_a)
    normalized_constant = sp.simplify(coefficient_c / coefficient_a)
    half_linear_coefficient = sp.simplify(normalized_linear_coefficient / 2)
    square_completion = sp.simplify(half_linear_coefficient**2)
    monic_denominator = sp.Add(
        variable**2,
        sp.Mul(normalized_linear_coefficient, variable, evaluate=False),
        normalized_constant,
        evaluate=False,
    )
    expanded_completion = sp.Add(
        variable**2,
        sp.Mul(normalized_linear_coefficient, variable, evaluate=False),
        square_completion,
        radius_squared,
        evaluate=False,
    )
    pattern_variable = sp.Symbol("z", real=True)
    pattern_linear = sp.Symbol("p", real=True)
    pattern_constant = sp.Symbol("q", real=True)
    generic_quadratic = sp.Add(
        pattern_variable**2,
        sp.Mul(pattern_linear, pattern_variable, evaluate=False),
        pattern_constant,
        evaluate=False,
    )
    generic_completed_quadratic = sp.Add(
        sp.Pow(
            sp.Add(pattern_variable, pattern_linear / 2, evaluate=False),
            2,
            evaluate=False,
        ),
        sp.Add(pattern_constant, -(pattern_linear**2 / 4), evaluate=False),
        evaluate=False,
    )
    notes: list[BackendMathNote] = []
    if str(coefficient_a) != "1":
        notes.append(
            BackendMathNote(
                label="First factor the leading coefficient",
                expression=BackendIdentity(
                    left=denominator,
                    right=sp.Mul(coefficient_a, monic_denominator, evaluate=False),
                ),
            )
        )
    notes.extend(
        (
            BackendMathNote(
                label="Take half the linear coefficient, then square it",
                expression=BackendIdentity(
                    left=sp.Pow(half_linear_coefficient, 2, evaluate=False),
                    right=square_completion,
                ),
            ),
            BackendMathNote(
                label="Add and subtract that number",
                expression=BackendIdentity(left=monic_denominator, right=expanded_completion),
            ),
            BackendMathNote(
                label="Recognize the perfect square",
                expression=BackendIdentity(left=monic_denominator, right=completed_denominator),
            ),
            BackendMathNote(
                label="General pattern",
                expression=BackendIdentity(
                    left=generic_quadratic,
                    right=generic_completed_quadratic,
                ),
            ),
        )
    )
    return tuple(notes)


def _build_reciprocal_quadratic_steps(
    context: _ReciprocalQuadraticDerivation,
) -> tuple[BackendDerivationStep, ...]:
    """Choose only the transformations that add pedagogical value."""
    integrand = context.integrand
    variable = context.variable
    denominator = context.denominator
    coefficient_a = context.coefficient_a
    coefficient_b = context.coefficient_b
    coefficient_c = context.coefficient_c
    radius_squared = context.radius_squared
    completed_denominator = context.completed_denominator
    completed_integrand = context.completed_integrand
    shift = context.shift
    radius = context.radius
    substitution_variable = context.substitution_variable
    unit_integrand = context.unit_integrand
    normalized_argument = context.normalized_argument
    normalized_coefficient = context.normalized_coefficient
    coefficient_is_one = context.coefficient_is_one
    formula_in_substitution_variable = context.formula_in_substitution_variable
    formula = context.formula
    integration_constant = context.integration_constant
    displayed_coefficient = None if coefficient_is_one else normalized_coefficient
    original_integral = BackendIntegral(integrand=integrand, variable=variable)
    completed_integral = BackendIntegral(integrand=completed_integrand, variable=variable)
    normalized_integral = BackendIntegral(
        integrand=unit_integrand,
        variable=substitution_variable,
        coefficient=displayed_coefficient,
    )
    needs_completion = coefficient_b != sp.Integer(0)
    needs_substitution = normalized_argument != variable or not coefficient_is_one
    steps: list[BackendDerivationStep] = []
    if needs_completion:
        steps.append(
            BackendDerivationStep(
                rule="Complete the square",
                before=original_integral,
                after=completed_integral,
                explanation=(
                    "Rewrite the quadratic denominator as a shifted square plus a positive "
                    "constant so it can match a standard integration rule."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between the two integrands gives zero."
                ),
                notes=_completion_notes(
                    denominator=denominator,
                    variable=variable,
                    coefficient_a=coefficient_a,
                    coefficient_b=coefficient_b,
                    coefficient_c=coefficient_c,
                    radius_squared=radius_squared,
                    completed_denominator=completed_denominator,
                ),
            )
        )
    if needs_substitution:
        scaled_substitution_variable = (
            substitution_variable
            if radius == sp.Integer(1)
            else sp.Mul(radius, substitution_variable, evaluate=False)
        )
        steps.append(
            BackendDerivationStep(
                rule="Substitute to get a unit denominator",
                before=completed_integral if needs_completion else original_integral,
                after=normalized_integral,
                explanation=(
                    "Scale the shifted variable so the denominator becomes one plus its square. "
                    "Transform the differential at the same time."
                ),
                verification_method=VerificationMethod.SUBSTITUTION,
                verification_detail=(
                    "Replacing the new variable and its differential recovers the previous "
                    "integral."
                ),
                notes=(
                    BackendMathNote(
                        label="Choose the substitution",
                        expression=sp.Eq(
                            substitution_variable,
                            normalized_argument,
                            evaluate=False,
                        ),
                    ),
                    BackendMathNote(
                        label="Rewrite the shifted term",
                        expression=BackendIdentity(
                            left=shift,
                            right=scaled_substitution_variable,
                        ),
                    ),
                    BackendMathNote(
                        label="Change the differential",
                        expression=BackendIdentity(
                            left=BackendDifferential(variable=variable),
                            right=BackendDifferential(
                                variable=substitution_variable,
                                coefficient=None if radius == sp.Integer(1) else radius,
                            ),
                        ),
                    ),
                ),
            )
        )
    steps.append(
        BackendDerivationStep(
            rule="Use the basic arctangent rule",
            before=normalized_integral if needs_substitution else original_integral,
            after=formula_in_substitution_variable if needs_substitution else formula,
            explanation=(
                "The remaining integral is the derivative pattern for the arctangent function."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the arctangent expression recovers the displayed integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=unit_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.Add(
                            sp.atan(substitution_variable),
                            integration_constant,
                            evaluate=False,
                        ),
                    ),
                ),
            ),
        )
    )
    if needs_substitution:
        steps.append(
            BackendDerivationStep(
                rule="Substitute back",
                before=formula_in_substitution_variable,
                after=formula,
                explanation=(
                    "Replace the temporary variable with its expression in the original variable."
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
                            normalized_argument,
                            evaluate=False,
                        ),
                    ),
                ),
            )
        )
    return tuple(steps)


def derive_reciprocal_quadratic_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a completed-square arctangent integral when applicable."""
    numerator, denominator = sp.fraction(sp.together(integrand))
    if numerator.has(variable):
        return ()
    polynomial = sp.Poly(denominator, variable)
    if polynomial.degree() != _QUADRATIC_DEGREE:
        return ()
    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    center = sp.simplify(-coefficient_b / (2 * coefficient_a))
    radius_squared = sp.simplify(
        coefficient_c / coefficient_a - coefficient_b**2 / (4 * coefficient_a**2)
    )
    if radius_squared.is_positive is not True:
        return ()
    prefactor = sp.simplify(numerator / coefficient_a)
    shift = variable if center == sp.Integer(0) else sp.Add(variable, -center, evaluate=False)
    completed_denominator = sp.Add(
        sp.Pow(shift, 2, evaluate=False),
        radius_squared,
        evaluate=False,
    )
    completed_integrand = sp.Mul(
        prefactor,
        sp.Pow(completed_denominator, -1, evaluate=False),
        evaluate=False,
    )
    if str(sp.simplify(integrand - completed_integrand)) != "0":
        message = "completing the square changed the integrand"
        raise ValueError(message)
    radius = sp.sqrt(radius_squared)
    radius_scale = sp.sqrt(sp.simplify(4 * radius_squared))
    raw_normalized_argument = sp.Mul(
        sp.expand(2 * shift),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    raw_normalized_coefficient = sp.Mul(
        sp.expand(2 * prefactor),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    if isinstance(radius_scale, sp.Rational):
        normalized_argument = sp.cancel(raw_normalized_argument)
        normalized_coefficient = sp.cancel(raw_normalized_coefficient)
    else:
        normalized_argument = raw_normalized_argument
        normalized_coefficient = raw_normalized_coefficient
    coefficient_is_one = sp.simplify(normalized_coefficient - 1) == sp.Integer(0)
    substitution_variable = sp.Symbol("u", real=True)
    unit_denominator = sp.Add(
        sp.Pow(substitution_variable, 2, evaluate=False),
        sp.Integer(1),
        evaluate=False,
    )
    unit_integrand = sp.Pow(unit_denominator, -1, evaluate=False)
    integration_constant = sp.Symbol("C")
    formula_in_substitution_variable_term = (
        sp.atan(substitution_variable)
        if coefficient_is_one
        else sp.Mul(
            normalized_coefficient,
            sp.atan(substitution_variable),
            evaluate=False,
        )
    )
    formula_in_substitution_variable = sp.Add(
        formula_in_substitution_variable_term,
        integration_constant,
        evaluate=False,
    )
    formula_term = (
        sp.atan(normalized_argument)
        if coefficient_is_one
        else sp.Mul(
            normalized_coefficient,
            sp.atan(normalized_argument),
            evaluate=False,
        )
    )
    formula = sp.Add(
        formula_term,
        integration_constant,
        evaluate=False,
    )
    transformed_integrand = sp.Mul(
        normalized_coefficient,
        unit_integrand.subs(substitution_variable, normalized_argument),
        sp.diff(normalized_argument, variable),
    )
    if str(sp.simplify(transformed_integrand - completed_integrand)) != "0":
        message = "the substitution changed the completed-square integral"
        raise ValueError(message)
    if (
        str(
            sp.simplify(
                sp.diff(formula_in_substitution_variable, substitution_variable)
                - normalized_coefficient * unit_integrand
            )
        )
        != "0"
    ):
        message = "the arctangent formula failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the simplified antiderivative differs from the exact result"
        raise ValueError(message)
    return _build_reciprocal_quadratic_steps(
        _ReciprocalQuadraticDerivation(
            integrand=integrand,
            variable=variable,
            denominator=denominator,
            coefficient_a=coefficient_a,
            coefficient_b=coefficient_b,
            coefficient_c=coefficient_c,
            radius_squared=radius_squared,
            completed_denominator=completed_denominator,
            completed_integrand=completed_integrand,
            shift=shift,
            radius=radius,
            substitution_variable=substitution_variable,
            unit_integrand=unit_integrand,
            normalized_argument=normalized_argument,
            normalized_coefficient=normalized_coefficient,
            coefficient_is_one=coefficient_is_one,
            formula_in_substitution_variable=formula_in_substitution_variable,
            formula=formula,
            integration_constant=integration_constant,
        )
    )
