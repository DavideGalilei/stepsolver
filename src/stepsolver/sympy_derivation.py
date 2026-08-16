"""Backend-local equation derivation strategies for human-readable steps."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.results import VerificationMethod

type BackendExpression = sp.Basic | tuple[sp.Basic, ...]

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


def _equivalent_step(
    *,
    rule: str,
    before: BackendExpression,
    after: BackendExpression,
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


def _solution_set(expression: BackendExpression, variable: sp.Symbol) -> frozenset[str]:
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
    before: BackendExpression,
    after: BackendExpression,
    explanation: str,
    variable: sp.Symbol,
) -> BackendExpression:
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
    current: BackendExpression = equation
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
    final: BackendExpression = (
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
    current: BackendExpression = equation
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
    current: BackendExpression = equation
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
