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


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDifferential:
    """A displayed differential such as dx or du."""

    variable: sp.Symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIdentity:
    """A displayed equality between two backend derivation expressions."""

    left: BackendExpression
    right: BackendExpression


type BackendExpression = (
    EquationBackendExpression | BackendIntegral | BackendDifferential | BackendIdentity
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
    arctangent_argument = sp.Mul(
        shift,
        sp.Pow(radius, -1, evaluate=False),
        evaluate=False,
    )
    raw_coefficient = sp.Mul(
        prefactor,
        sp.Pow(radius, -1, evaluate=False),
        evaluate=False,
    )
    formula = sp.Mul(
        raw_coefficient,
        sp.atan(arctangent_argument),
        evaluate=False,
    )
    integration_constant = sp.Symbol("C")
    formula = sp.Add(formula, integration_constant, evaluate=False)
    if str(sp.simplify(sp.diff(formula, variable) - completed_integrand)) != "0":
        message = "the arctangent formula failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the simplified antiderivative differs from the exact result"
        raise ValueError(message)
    completed_polynomial = (
        completed_denominator
        if str(coefficient_a) == "1"
        else sp.Mul(coefficient_a, completed_denominator, evaluate=False)
    )
    generic_coefficient_a = sp.Symbol("a", real=True)
    generic_coefficient_b = sp.Symbol("b", real=True)
    generic_coefficient_c = sp.Symbol("c", real=True)
    generic_quadratic = sp.Add(
        sp.Mul(generic_coefficient_a, variable**2, evaluate=False),
        sp.Mul(generic_coefficient_b, variable, evaluate=False),
        generic_coefficient_c,
        evaluate=False,
    )
    generic_shift = sp.Add(
        variable,
        generic_coefficient_b / (2 * generic_coefficient_a),
        evaluate=False,
    )
    generic_completed_quadratic = sp.Add(
        sp.Mul(
            generic_coefficient_a,
            sp.Pow(generic_shift, 2, evaluate=False),
            evaluate=False,
        ),
        sp.Add(
            generic_coefficient_c,
            -(generic_coefficient_b**2 / (4 * generic_coefficient_a)),
            evaluate=False,
        ),
        evaluate=False,
    )
    generic_variable = sp.Symbol("u", real=True)
    generic_radius = sp.Symbol("a", real=True)
    generic_integrand = sp.Pow(
        sp.Add(
            sp.Pow(generic_variable, 2, evaluate=False),
            sp.Pow(generic_radius, 2, evaluate=False),
            evaluate=False,
        ),
        -1,
        evaluate=False,
    )
    generic_result = sp.Add(
        sp.Mul(
            sp.Pow(generic_radius, -1, evaluate=False),
            sp.atan(
                sp.Mul(
                    generic_variable,
                    sp.Pow(generic_radius, -1, evaluate=False),
                    evaluate=False,
                )
            ),
            evaluate=False,
        ),
        integration_constant,
        evaluate=False,
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
            notes=(
                BackendMathNote(
                    label="Standard identity",
                    expression=BackendIdentity(
                        left=generic_quadratic,
                        right=generic_completed_quadratic,
                    ),
                ),
                BackendMathNote(
                    label="Coefficients in this denominator",
                    expression=(
                        sp.Eq(generic_coefficient_a, coefficient_a, evaluate=False),
                        sp.Eq(generic_coefficient_b, coefficient_b, evaluate=False),
                        sp.Eq(generic_coefficient_c, coefficient_c, evaluate=False),
                    ),
                ),
                BackendMathNote(
                    label="Applied to this denominator",
                    expression=BackendIdentity(
                        left=denominator,
                        right=completed_polynomial,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the arctangent integral",
            before=BackendIntegral(integrand=completed_integrand, variable=variable),
            after=formula,
            explanation="Match the completed denominator to the standard arctangent form.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating this arctangent expression recovers "
                "the completed-square integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Standard rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=generic_integrand,
                            variable=generic_variable,
                        ),
                        right=generic_result,
                    ),
                ),
                BackendMathNote(
                    label="Match the variables",
                    expression=(
                        sp.Eq(generic_variable, shift, evaluate=False),
                        sp.Eq(generic_radius, radius, evaluate=False),
                    ),
                ),
                BackendMathNote(
                    label="Differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=generic_variable),
                        right=BackendDifferential(variable=variable),
                    ),
                ),
            ),
        ),
    ]
    if str(formula) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Simplify the antiderivative",
                before=formula,
                after=result,
                explanation="Simplify the constant factor and the arctangent argument.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between both antiderivative forms gives zero."
                ),
                notes=(
                    BackendMathNote(
                        label="Simplify the coefficient",
                        expression=sp.Eq(
                            raw_coefficient,
                            sp.simplify(raw_coefficient),
                            evaluate=False,
                        ),
                    ),
                    BackendMathNote(
                        label="Simplify the argument",
                        expression=sp.Eq(
                            arctangent_argument,
                            sp.simplify(arctangent_argument),
                            evaluate=False,
                        ),
                    ),
                ),
            )
        )
    return tuple(steps)
