"""Backend-local derivation strategies for human-readable solution steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.results import VerificationMethod

type EquationBackendExpression = sp.Basic | tuple[sp.Basic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIntegral:
    """An unevaluated backend integral used only for derivation display."""

    integrand: sp.Basic
    variable: sp.Symbol
    coefficient: sp.Basic | None = None
    lower: sp.Basic | None = None
    upper: sp.Basic | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDifferential:
    """A displayed differential such as dx or du."""

    variable: sp.Symbol
    coefficient: sp.Basic | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDerivative:
    """A displayed derivative of a backend expression."""

    expression: sp.Basic
    variable: sp.Symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendLimit:
    """A displayed one- or two-sided limit."""

    expression: sp.Basic
    variable: sp.Symbol
    point: sp.Basic
    direction: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIdentity:
    """A displayed equality between two backend derivation expressions."""

    left: BackendExpression
    right: BackendExpression


type BackendExpression = (
    EquationBackendExpression
    | BackendIntegral
    | BackendDifferential
    | BackendDerivative
    | BackendLimit
    | BackendIdentity
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendMathNote:
    """A labeled mathematical annotation supporting a derivation step."""

    label: str
    expression: BackendExpression


_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2


def _is_basic_sequence(value: object) -> TypeGuard[Sequence[sp.Basic]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    items = cast("Sequence[object]", value)
    return all(isinstance(item, sp.Basic) for item in items)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDerivationStep:
    """One backend-native transformation awaiting conversion to the public AST."""

    rule: str
    before: BackendExpression
    after: BackendExpression
    explanation: str
    verification_method: VerificationMethod
    verification_detail: str
    notes: tuple[BackendMathNote, ...] = ()


def _equivalent_step(
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
) -> BackendDerivationStep:
    if _solution_set(before, variable) != _solution_set(after, variable):
        message = "a proposed derivation step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail="Both forms have the same solution set for the target variable.",
    )


def _solution_set(
    expression: EquationBackendExpression,
    variable: sp.Symbol,
) -> frozenset[str]:
    equations = expression if isinstance(expression, tuple) else (expression,)
    roots: set[str] = set()
    for equation in equations:
        solved = sp.solve(equation, variable)
        if not _is_basic_sequence(solved):
            message = "equation verification did not produce a root sequence"
            raise TypeError(message)
        roots.update(str(sp.simplify(root)) for root in solved)
    return frozenset(roots)


def _relations(variable: sp.Symbol, roots: tuple[sp.Basic, ...]) -> tuple[sp.Basic, ...]:
    return tuple(sp.Eq(variable, root) for root in roots)


def _append_if_changed(
    steps: list[BackendDerivationStep],
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
) -> EquationBackendExpression:
    if str(before) == str(after):
        return before
    steps.append(
        _equivalent_step(
            rule=rule,
            before=before,
            after=after,
            explanation=explanation,
            variable=variable,
        )
    )
    return after


def _linear_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    coefficient = polynomial.coeff_monomial(variable)
    constant = polynomial.coeff_monomial(1)
    isolated_term = sp.Eq(coefficient * variable, -constant)
    current: EquationBackendExpression = equation
    current = _append_if_changed(
        steps,
        rule="Collect variable terms",
        before=current,
        after=isolated_term,
        explanation=(
            "Move every variable term to one side and every constant term to the other side."
        ),
        variable=variable,
    )
    final: EquationBackendExpression = (
        sp.Eq(variable, roots[0]) if len(roots) == _LINEAR_DEGREE else _relations(variable, roots)
    )
    _append_if_changed(
        steps,
        rule="Divide by the coefficient",
        before=current,
        after=final,
        explanation="Divide both sides by the coefficient of the variable.",
        variable=variable,
    )
    return tuple(steps)


def _quadratic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    expression = polynomial.as_expr()
    factored = sp.factor(expression)
    current: EquationBackendExpression = equation
    if str(factored) != str(expression) and factored.is_Mul:
        factored_equation = sp.Eq(factored, 0)
        current = _append_if_changed(
            steps,
            rule="Factor the quadratic",
            before=current,
            after=factored_equation,
            explanation="Rewrite the quadratic as a product of linear factors.",
            variable=variable,
        )
        factors = tuple(
            sp.Eq(factor, 0) for factor in factored.as_ordered_factors() if factor.has(variable)
        )
        current = _append_if_changed(
            steps,
            rule="Apply the zero-product property",
            before=current,
            after=factors,
            explanation="A product is zero only when at least one of its factors is zero.",
            variable=variable,
        )
        factor_roots = tuple(
            -sp.Poly(factor, variable).coeff_monomial(1)
            / sp.Poly(factor, variable).coeff_monomial(variable)
            for factor in factored.as_ordered_factors()
            if factor.has(variable)
        )
        _append_if_changed(
            steps,
            rule="Solve each factor",
            before=current,
            after=_relations(variable, factor_roots),
            explanation="Solve each resulting linear equation and combine the solutions.",
            variable=variable,
        )
        return tuple(steps)

    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    discriminant = sp.expand(coefficient_b**2 - 4 * coefficient_a * coefficient_c)
    denominator = 2 * coefficient_a
    negative_numerator = sp.Add(-coefficient_b, -sp.sqrt(discriminant), evaluate=False)
    positive_numerator = sp.Add(-coefficient_b, sp.sqrt(discriminant), evaluate=False)
    formula_roots = tuple(
        sp.Mul(numerator, denominator**-1, evaluate=False)
        for numerator in (negative_numerator, positive_numerator)
    )
    current = _append_if_changed(
        steps,
        rule="Apply the quadratic formula",
        before=current,
        after=_relations(variable, formula_roots),
        explanation=("Use x = (-b ± √(b² - 4ac)) / (2a) with the equation's coefficients."),
        variable=variable,
    )
    _append_if_changed(
        steps,
        rule="Simplify the roots",
        before=current,
        after=_relations(variable, roots),
        explanation="Simplify the discriminant and both resulting exact roots.",
        variable=variable,
    )
    return tuple(steps)


def derive_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive detailed steps for rational, linear, and quadratic equations."""
    steps: list[BackendDerivationStep] = []
    difference = sp.together(equation.lhs - equation.rhs)
    numerator, denominator = sp.fraction(difference)
    expanded = sp.expand(numerator)
    normalized = sp.Eq(expanded, 0)
    current: EquationBackendExpression = equation
    if str(denominator) != "1":
        current = _append_if_changed(
            steps,
            rule="Clear the denominators",
            before=current,
            after=normalized,
            explanation=("Multiply through by the common denominator, which must be nonzero."),
            variable=variable,
        )
    polynomial = sp.Poly(expanded, variable)
    degree = polynomial.degree()
    if degree == _LINEAR_DEGREE:
        detail = _linear_steps(normalized if steps else equation, variable, polynomial, roots)
    elif degree == _QUADRATIC_DEGREE:
        normalized_current = normalized
        if not steps:
            current = _append_if_changed(
                steps,
                rule="Write in standard form",
                before=current,
                after=normalized,
                explanation="Move every term to one side and combine like terms.",
                variable=variable,
            )
            if isinstance(current, sp.Equality):
                normalized_current = current
        detail = _quadratic_steps(normalized_current, variable, polynomial, roots)
    else:
        return ()
    steps.extend(detail)
    return tuple(steps)


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
    shift = sp.Add(variable, -center, evaluate=False)
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
    normalized_argument = sp.Mul(
        sp.expand(2 * shift),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    normalized_coefficient = sp.Mul(
        sp.expand(2 * prefactor),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    substitution_variable = sp.Symbol("u", real=True)
    unit_denominator = sp.Add(
        sp.Pow(substitution_variable, 2, evaluate=False),
        sp.Integer(1),
        evaluate=False,
    )
    unit_integrand = sp.Pow(unit_denominator, -1, evaluate=False)
    integration_constant = sp.Symbol("C")
    formula_in_substitution_variable = sp.Add(
        sp.Mul(
            normalized_coefficient,
            sp.atan(substitution_variable),
            evaluate=False,
        ),
        integration_constant,
        evaluate=False,
    )
    formula = sp.Add(
        sp.Mul(
            normalized_coefficient,
            sp.atan(normalized_argument),
            evaluate=False,
        ),
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
    if str(
        sp.simplify(
            sp.diff(formula_in_substitution_variable, substitution_variable)
            - normalized_coefficient * unit_integrand
        )
    ) != "0":
        message = "the arctangent formula failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the simplified antiderivative differs from the exact result"
        raise ValueError(message)
    completion_notes = _completion_notes(
        denominator=denominator,
        variable=variable,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        coefficient_c=coefficient_c,
        radius_squared=radius_squared,
        completed_denominator=completed_denominator,
    )
    steps = [
        BackendDerivationStep(
            rule="Complete the square",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=completed_integrand, variable=variable),
            explanation=(
                "Rewrite the quadratic denominator as a shifted square plus a positive constant "
                "so it can match a standard integration rule."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Simplifying the difference between the two integrands gives zero.",
            notes=completion_notes,
        ),
        BackendDerivationStep(
            rule="Substitute to get a unit denominator",
            before=BackendIntegral(integrand=completed_integrand, variable=variable),
            after=BackendIntegral(
                integrand=unit_integrand,
                variable=substitution_variable,
                coefficient=normalized_coefficient,
            ),
            explanation=(
                "Scale the shifted variable so the denominator becomes one plus its square. "
                "Transform the differential at the same time."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the new variable and its differential recovers the previous integral."
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
                        right=sp.Mul(radius, substitution_variable, evaluate=False),
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=substitution_variable,
                            coefficient=radius,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the basic arctangent rule",
            before=BackendIntegral(
                integrand=unit_integrand,
                variable=substitution_variable,
                coefficient=normalized_coefficient,
            ),
            after=formula_in_substitution_variable,
            explanation=(
                "The remaining integral is the derivative pattern for the arctangent function."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the arctangent expression with respect to the new variable "
                "recovers the transformed integrand."
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
        ),
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
        ),
    ]
    return tuple(steps)


def derive_dirichlet_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the classical Dirichlet integral with a damping parameter."""
    expected_integrand = sp.sin(variable) / variable
    if str(sp.simplify(integrand - expected_integrand)) != "0":
        return ()
    if str(sp.simplify(lower)) != "0" or upper != sp.oo:
        return ()
    if str(sp.simplify(result - sp.pi / 2)) != "0":
        return ()

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
