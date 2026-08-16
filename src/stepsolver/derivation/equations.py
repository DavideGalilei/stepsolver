"""Human-readable derivations for polynomial equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendQuadraticSolutions,
    BackendStepConstraint,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod

_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2
_CONSTANT_DEGREE = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _DomainRestrictions:
    denominators: tuple[sp.Basic, ...]
    excluded_values: tuple[sp.Basic, ...]
    displayed: tuple[BackendStepConstraint, ...]


def _is_basic_sequence(value: object) -> TypeGuard[Sequence[sp.Basic]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    items = cast("Sequence[object]", value)
    return all(isinstance(item, sp.Basic) for item in items)


def _equivalent_step(
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
) -> BackendDerivationStep:
    if _solution_set(before, variable, excluded_values) != _solution_set(
        after, variable, excluded_values
    ):
        message = "a proposed derivation step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "Both forms have the same solution set under the stated domain restrictions."
            if excluded_values
            else "Both forms have the same solution set for the target variable."
        ),
        introduced_constraints=introduced_constraints,
    )


def _solution_set(
    expression: EquationBackendExpression,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
) -> frozenset[str]:
    equations = expression if isinstance(expression, tuple) else (expression,)
    roots: set[str] = set()
    for equation in equations:
        solved = sp.solve(equation, variable)
        if not _is_basic_sequence(solved):
            message = "equation verification did not produce a root sequence"
            raise TypeError(message)
        roots.update(
            str(sp.simplify(root))
            for root in solved
            if not any(
                sp.simplify(root - excluded) == sp.Integer(0) for excluded in excluded_values
            )
        )
    return frozenset(roots)


def _relations(variable: sp.Symbol, roots: tuple[sp.Basic, ...]) -> tuple[sp.Basic, ...]:
    return tuple(sp.Eq(variable, root) for root in roots)


def _domain_restrictions(
    denominator: sp.Basic,
    supplied_denominators: tuple[sp.Basic, ...],
    variable: sp.Symbol,
) -> _DomainRestrictions:
    denominators = list(supplied_denominators)
    if not denominators and denominator != sp.Integer(1):
        denominators.append(denominator)
    unique_text = tuple(dict.fromkeys(str(item) for item in denominators))
    unique_denominators = tuple(
        next(item for item in denominators if str(item) == text) for text in unique_text
    )
    exclusions: list[sp.Basic] = []
    for domain_denominator in unique_denominators:
        solved = sp.solve(domain_denominator, variable)
        if _is_basic_sequence(solved):
            exclusions.extend(solved)
    excluded_values = tuple(dict.fromkeys(exclusions))
    denominator_constraints = tuple(
        BackendStepConstraint(
            explanation="An original denominator cannot equal zero.",
            expression=BackendNotEqual(left=item, right=sp.Integer(0)),
        )
        for item in unique_denominators
    )
    exclusion_constraints = tuple(
        BackendStepConstraint(
            explanation="This value is outside the domain of the original equation.",
            expression=BackendNotEqual(left=variable, right=value),
        )
        for value in excluded_values
    )
    return _DomainRestrictions(
        denominators=unique_denominators,
        excluded_values=excluded_values,
        displayed=denominator_constraints + exclusion_constraints,
    )


def _append_if_changed(
    steps: list[BackendDerivationStep],
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
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
            excluded_values=excluded_values,
            introduced_constraints=introduced_constraints,
        )
    )
    return after


def _linear_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
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
        excluded_values=excluded_values,
    )
    candidate = sp.simplify(-constant / coefficient)
    candidate_relation = sp.Eq(variable, candidate)
    current = _append_if_changed(
        steps,
        rule="Divide by the coefficient",
        before=current,
        after=candidate_relation,
        explanation="Divide both sides by the coefficient of the variable.",
        variable=variable,
        excluded_values=excluded_values,
    )
    if not roots:
        _append_if_changed(
            steps,
            rule="Apply the domain restriction",
            before=current,
            after=(),
            explanation=(
                "The algebraic candidate makes an original denominator zero, so reject it."
            ),
            variable=variable,
            excluded_values=excluded_values,
        )
    return tuple(steps)


def _quadratic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    expression = polynomial.as_expr()
    factored = sp.factor(expression)
    current: EquationBackendExpression = equation
    if str(factored) != str(expression) and (factored.is_Mul or factored.is_Pow):
        _, factor_pairs = sp.factor_list(expression, variable)
        factor_bases = tuple(
            factor for factor, _multiplicity in factor_pairs if factor.has(variable)
        )
        has_nonlinear_factor = any(
            sp.Poly(factor, variable).degree() != 1 for factor in factor_bases
        )
        if not factor_bases or has_nonlinear_factor:
            return ()
        factored_equation = sp.Eq(factored, 0)
        current = _append_if_changed(
            steps,
            rule="Factor the quadratic",
            before=current,
            after=factored_equation,
            explanation="Rewrite the quadratic as a product of linear factors.",
            variable=variable,
            excluded_values=excluded_values,
        )
        factor_equations = tuple(sp.Eq(factor, 0) for factor in factor_bases)
        separated_factors: EquationBackendExpression = (
            factor_equations[0] if len(factor_equations) == 1 else factor_equations
        )
        current = _append_if_changed(
            steps,
            rule=(
                "Set the repeated factor equal to zero"
                if len(factor_equations) == 1
                else "Apply the zero-product property"
            ),
            before=current,
            after=separated_factors,
            explanation=(
                "A power is zero only when its base is zero."
                if len(factor_equations) == 1
                else "A product is zero only when at least one of its factors is zero."
            ),
            variable=variable,
            excluded_values=excluded_values,
        )
        factor_roots = tuple(
            -sp.Poly(factor, variable).coeff_monomial(1)
            / sp.Poly(factor, variable).coeff_monomial(variable)
            for factor in factor_bases
        )
        current = _append_if_changed(
            steps,
            rule="Solve each factor",
            before=current,
            after=_relations(variable, factor_roots),
            explanation="Solve each resulting linear equation and combine the solutions.",
            variable=variable,
            excluded_values=excluded_values,
        )
        if set(map(str, factor_roots)) != set(map(str, roots)):
            _append_if_changed(
                steps,
                rule="Apply the domain restrictions",
                before=current,
                after=_relations(variable, roots),
                explanation=(
                    "Discard every candidate that makes an original denominator equal to zero."
                ),
                variable=variable,
                excluded_values=excluded_values,
            )
        return tuple(steps)

    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    discriminant = sp.expand(coefficient_b**2 - 4 * coefficient_a * coefficient_c)
    if not roots and discriminant.is_negative is True:
        discriminant_symbol = sp.Symbol("delta", real=True)
        discriminant_value = sp.Eq(discriminant_symbol, discriminant, evaluate=False)
        generic_discriminant = sp.Eq(
            discriminant_symbol,
            sp.Symbol("b") ** 2 - 4 * sp.Symbol("a") * sp.Symbol("c"),
            evaluate=False,
        )
        return (
            BackendDerivationStep(
                rule="Calculate the discriminant",
                before=equation,
                after=discriminant_value,
                explanation="Compute the discriminant to determine the type of roots.",
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="The discriminant was evaluated from the exact coefficients.",
                notes=(
                    BackendMathNote(
                        label="Discriminant rule",
                        expression=generic_discriminant,
                    ),
                    BackendMathNote(
                        label="Coefficients",
                        expression=(
                            sp.Eq(sp.Symbol("a"), coefficient_a, evaluate=False),
                            sp.Eq(sp.Symbol("b"), coefficient_b, evaluate=False),
                            sp.Eq(sp.Symbol("c"), coefficient_c, evaluate=False),
                        ),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Conclude there are no real solutions",
                before=discriminant_value,
                after=(),
                explanation=(
                    "A negative discriminant means the quadratic has no roots in the real "
                    "number system."
                ),
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="Solving over the real numbers produces the empty set.",
            ),
        )
    denominator = 2 * coefficient_a
    negative_numerator = sp.Add(-coefficient_b, -sp.sqrt(discriminant), evaluate=False)
    positive_numerator = sp.Add(-coefficient_b, sp.sqrt(discriminant), evaluate=False)
    generic_a = sp.Symbol("a", nonzero=True)
    generic_b = sp.Symbol("b")
    generic_c = sp.Symbol("c")
    generic_formula_discriminant = generic_b**2 - 4 * generic_a * generic_c
    steps.append(
        BackendDerivationStep(
            rule="Apply the quadratic formula",
            before=current,
            after=BackendQuadraticSolutions(
                variable=variable,
                negative_numerator=negative_numerator,
                positive_numerator=positive_numerator,
                denominator=denominator,
            ),
            explanation=(
                "Identify a, b, and c, calculate the discriminant, then substitute into the "
                "quadratic formula."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The displayed quadratic-formula roots match the exact roots.",
            notes=(
                BackendMathNote(
                    label="Quadratic formula",
                    expression=BackendQuadraticSolutions(
                        variable=variable,
                        negative_numerator=sp.Add(
                            -generic_b,
                            -sp.sqrt(generic_formula_discriminant),
                            evaluate=False,
                        ),
                        positive_numerator=sp.Add(
                            -generic_b,
                            sp.sqrt(generic_formula_discriminant),
                            evaluate=False,
                        ),
                        denominator=2 * generic_a,
                    ),
                ),
                BackendMathNote(
                    label="Coefficient a",
                    expression=BackendIdentity(left=generic_a, right=coefficient_a),
                ),
                BackendMathNote(
                    label="Coefficient b",
                    expression=BackendIdentity(left=generic_b, right=coefficient_b),
                ),
                BackendMathNote(
                    label="Coefficient c",
                    expression=BackendIdentity(left=generic_c, right=coefficient_c),
                ),
                BackendMathNote(
                    label="Discriminant",
                    expression=BackendIdentity(
                        left=sp.Symbol("delta"),
                        right=discriminant,
                    ),
                ),
            ),
        )
    )
    return tuple(steps)


def derive_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
    domain_denominators: tuple[sp.Basic, ...] = (),
) -> tuple[BackendDerivationStep, ...]:
    """Derive detailed steps for rational, linear, and quadratic equations."""
    steps: list[BackendDerivationStep] = []
    difference = sp.together(equation.lhs - equation.rhs)
    numerator, denominator = sp.fraction(difference)
    restrictions = _domain_restrictions(denominator, domain_denominators, variable)
    excluded_values = restrictions.excluded_values
    expanded = sp.expand(numerator)
    normalized = sp.Eq(expanded, 0, evaluate=False)
    current: EquationBackendExpression = equation
    if restrictions.denominators:
        current = _append_if_changed(
            steps,
            rule="Clear the denominators",
            before=current,
            after=normalized,
            explanation=("Multiply through by the common denominator, which must be nonzero."),
            variable=variable,
            excluded_values=excluded_values,
            introduced_constraints=restrictions.displayed,
        )
    polynomial = sp.Poly(expanded, variable)
    degree = polynomial.degree()
    if degree == _CONSTANT_DEGREE:
        current = _append_if_changed(
            steps,
            rule="Simplify the equation",
            before=current,
            after=normalized,
            explanation="Combine like terms on both sides.",
            variable=variable,
            excluded_values=excluded_values,
        )
        steps.append(
            BackendDerivationStep(
                rule="Conclude there are no solutions",
                before=current,
                after=(),
                explanation="A nonzero constant cannot equal zero.",
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The simplified contradiction has an empty solution set.",
            )
        )
        return tuple(steps)
    if degree == _LINEAR_DEGREE:
        detail = _linear_steps(
            normalized if steps else equation,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    elif degree == _QUADRATIC_DEGREE:
        normalized_current = normalized
        if not steps:
            if sp.simplify(equation.rhs) == sp.Integer(0):
                normalized_current = equation
            else:
                current = _append_if_changed(
                    steps,
                    rule="Write in standard form",
                    before=current,
                    after=normalized,
                    explanation="Move every term to one side and combine like terms.",
                    variable=variable,
                    excluded_values=excluded_values,
                )
                if isinstance(current, sp.Equality):
                    normalized_current = current
        detail = _quadratic_steps(
            normalized_current,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    else:
        return ()
    steps.extend(detail)
    return tuple(steps)
