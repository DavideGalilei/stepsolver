"""Human-readable derivations for polynomial equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendApproximateSolutions,
    BackendCardanoSolution,
    BackendCrossedOut,
    BackendDerivationStep,
    BackendExpression,
    BackendIdentity,
    BackendMathNote,
    BackendNewtonRule,
    BackendNotEqual,
    BackendProduct,
    BackendQuadraticSolutions,
    BackendQuotient,
    BackendStepConstraint,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import is_real_expression

_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2
_CUBIC_DEGREE = 3
_CONSTANT_DEGREE = 0
_ROOT_VERIFICATION_DIGITS = 12


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


def _equivalent_display_step(
    *,
    rule: str,
    semantic_before: EquationBackendExpression,
    semantic_after: EquationBackendExpression,
    display_before: BackendExpression,
    display_after: BackendExpression,
    explanation: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...] = (),
    introduced_constraints: tuple[BackendStepConstraint, ...] = (),
    verification_detail: str = "Both displayed forms preserve the same solution set.",
) -> BackendDerivationStep:
    if _solution_set(semantic_before, variable, excluded_values) != _solution_set(
        semantic_after, variable, excluded_values
    ):
        message = "a proposed display step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=display_before,
        after=display_after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=verification_detail,
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
            (
                str(sp.simplify(root))
                if root.free_symbols
                else str(sp.N(root, _ROOT_VERIFICATION_DIGITS))
            )
            for root in solved
            if is_real_expression(root)
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


def _cleared_side(expression: sp.Basic, denominator: sp.Basic) -> sp.Basic:
    if expression == sp.Integer(0):
        return expression
    _numerator, expression_denominator = sp.fraction(sp.together(expression))
    product = expression * denominator
    if expression_denominator != sp.Integer(1):
        return sp.cancel(product)
    return sp.Mul(expression, denominator, evaluate=False)


def _multiplied_side(expression: sp.Basic, denominator: sp.Basic) -> sp.Basic:
    if expression == sp.Integer(0):
        return expression
    return sp.Mul(denominator, expression, evaluate=False)


def _cancelled_side(expression: sp.Basic, denominator: sp.Basic) -> BackendExpression:
    if expression == sp.Integer(0):
        return expression
    numerator, expression_denominator = sp.fraction(sp.together(expression))
    if expression_denominator == sp.Integer(1):
        return BackendProduct(factors=(denominator, expression))
    remaining_multiplier = sp.cancel(denominator / expression_denominator)
    numerator_factors: list[BackendExpression] = []
    if remaining_multiplier != sp.Integer(1):
        numerator_factors.append(remaining_multiplier)
    numerator_factors.append(BackendCrossedOut(expression=expression_denominator))
    if numerator != sp.Integer(1):
        numerator_factors.append(numerator)
    displayed_numerator: BackendExpression
    if len(numerator_factors) == 1:
        displayed_numerator = numerator_factors[0]
    else:
        displayed_numerator = BackendProduct(factors=tuple(numerator_factors))
    return BackendQuotient(
        numerator=displayed_numerator,
        denominator=BackendCrossedOut(expression=expression_denominator),
    )


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


def _factored_polynomial_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    expression = polynomial.as_expr()
    factored = sp.factor(expression)
    if str(factored) == str(expression) or not (factored.is_Mul or factored.is_Pow):
        return ()
    _coefficient, factor_pairs = sp.factor_list(expression, variable)
    factor_bases = tuple(factor for factor, _multiplicity in factor_pairs if factor.has(variable))
    if not factor_bases:
        return ()
    steps: list[BackendDerivationStep] = []
    factored_equation = sp.Eq(factored, 0, evaluate=False)
    current: EquationBackendExpression = _append_if_changed(
        steps,
        rule="Factor the polynomial",
        before=equation,
        after=factored_equation,
        explanation="Rewrite the polynomial as a product of lower-degree factors.",
        variable=variable,
        excluded_values=excluded_values,
    )
    factor_equations = tuple(sp.Eq(factor, 0, evaluate=False) for factor in factor_bases)
    separated: EquationBackendExpression = (
        factor_equations[0] if len(factor_equations) == 1 else factor_equations
    )
    current = _append_if_changed(
        steps,
        rule="Apply the zero-product property",
        before=current,
        after=separated,
        explanation="At least one factor must equal zero, so solve each factor separately.",
        variable=variable,
        excluded_values=excluded_values,
    )
    _append_if_changed(
        steps,
        rule="Keep the real solutions",
        before=current,
        after=_relations(variable, roots),
        explanation=(
            "Solve the lower-degree equations and keep the real values allowed by the "
            "original domain."
        ),
        variable=variable,
        excluded_values=excluded_values,
    )
    return tuple(steps)


def _cubic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    factored = _factored_polynomial_steps(
        equation,
        variable,
        polynomial,
        roots,
        excluded_values,
    )
    if factored:
        return factored
    coefficient_a = polynomial.coeff_monomial(variable**3)
    coefficient_b = polynomial.coeff_monomial(variable**2)
    coefficient_c = polynomial.coeff_monomial(variable)
    coefficient_d = polynomial.coeff_monomial(1)
    depressed_linear = sp.simplify(
        (3 * coefficient_a * coefficient_c - coefficient_b**2) / (3 * coefficient_a**2)
    )
    depressed_constant = sp.simplify(
        (
            2 * coefficient_b**3
            - 9 * coefficient_a * coefficient_b * coefficient_c
            + 27 * coefficient_a**2 * coefficient_d
        )
        / (27 * coefficient_a**3)
    )
    discriminant = sp.simplify((depressed_constant / 2) ** 2 + (depressed_linear / 3) ** 3)
    if discriminant.is_positive is not True or len(roots) != 1:
        return ()
    reduced_variable = sp.Symbol("t", real=True)
    shift = sp.simplify(-coefficient_b / (3 * coefficient_a))
    depressed_equation = sp.Eq(
        reduced_variable**3 + depressed_linear * reduced_variable + depressed_constant,
        0,
        evaluate=False,
    )
    first_radicand = sp.simplify(-depressed_constant / 2 + sp.sqrt(discriminant))
    second_radicand = sp.simplify(-depressed_constant / 2 - sp.sqrt(discriminant))
    generic_p = sp.Symbol("p", real=True)
    generic_q = sp.Symbol("q", real=True)
    generic_a = sp.Symbol("a", real=True, nonzero=True)
    generic_b = sp.Symbol("b", real=True)
    generic_discriminant = (generic_q / 2) ** 2 + (generic_p / 3) ** 3
    cardano_solution = BackendCardanoSolution(
        variable=variable,
        shift=shift,
        first_radicand=first_radicand,
        second_radicand=second_radicand,
    )
    return (
        BackendDerivationStep(
            rule="Depress the cubic",
            before=equation,
            after=depressed_equation,
            explanation=(
                "Shift the variable to remove the squared term, producing the standard "
                "form t^3 + pt + q = 0."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Substituting the displayed variable shift gives this cubic.",
            notes=(
                BackendMathNote(
                    label="General substitution",
                    expression=BackendIdentity(
                        left=variable,
                        right=reduced_variable - generic_b / (3 * generic_a),
                    ),
                ),
                BackendMathNote(
                    label="For this cubic",
                    expression=BackendIdentity(
                        left=variable,
                        right=reduced_variable + shift,
                    ),
                ),
                BackendMathNote(
                    label="Coefficient p",
                    expression=BackendIdentity(left=generic_p, right=depressed_linear),
                ),
                BackendMathNote(
                    label="Coefficient q",
                    expression=BackendIdentity(left=generic_q, right=depressed_constant),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Apply Cardano's formula",
            before=depressed_equation,
            after=cardano_solution,
            explanation=(
                "The Cardano discriminant is positive, so the cubic has one real root. "
                "Substitute p and q into the real-root formula."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="Substitution of the displayed radical expression gives zero.",
            notes=(
                BackendMathNote(
                    label="Discriminant formula",
                    expression=BackendIdentity(
                        left=sp.Symbol("Delta"),
                        right=generic_discriminant,
                    ),
                ),
                BackendMathNote(
                    label="For this cubic",
                    expression=BackendIdentity(
                        left=sp.Symbol("Delta"),
                        right=discriminant,
                    ),
                ),
                BackendMathNote(
                    label="Real-root formula",
                    expression=BackendCardanoSolution(
                        variable=reduced_variable,
                        shift=sp.Integer(0),
                        first_radicand=-generic_q / 2 + sp.sqrt(generic_discriminant),
                        second_radicand=-generic_q / 2 - sp.sqrt(generic_discriminant),
                    ),
                ),
                BackendMathNote(
                    label="Decimal check",
                    expression=BackendIdentity(left=variable, right=sp.N(roots[0], 7)),
                ),
            ),
        ),
    )


def _generic_polynomial_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    factored = _factored_polynomial_steps(
        equation,
        variable,
        polynomial,
        roots,
        excluded_values,
    )
    if factored:
        return factored
    if not roots:
        return (
            _equivalent_step(
                rule="Check for real roots",
                before=equation,
                after=(),
                explanation=(
                    "The polynomial does not cross zero on the real number line, so it has "
                    "no real solutions."
                ),
                variable=variable,
                excluded_values=excluded_values,
            ),
        )
    approximations = tuple(sp.N(root, 7) for root in roots)
    return (
        BackendDerivationStep(
            rule="Approximate the real roots",
            before=equation,
            after=BackendApproximateSolutions(variable=variable, roots=approximations),
            explanation=(
                "The polynomial has no simpler exact factorization. Bracket each real root, "
                "then refine it with Newton's method."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Substitution gives a residual consistent with the displayed precision; "
                "the exact algebraic roots remain in the result."
            ),
            notes=(BackendMathNote(label="Newton iteration", expression=BackendNewtonRule()),),
        ),
    )


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
        multiplied = sp.Eq(
            _multiplied_side(equation.lhs, denominator),
            _multiplied_side(equation.rhs, denominator),
            evaluate=False,
        )
        multiplied_display = BackendIdentity(
            left=BackendProduct(factors=(denominator, equation.lhs)),
            right=BackendProduct(factors=(denominator, equation.rhs)),
        )
        cleared = sp.Eq(
            _cleared_side(equation.lhs, denominator),
            _cleared_side(equation.rhs, denominator),
            evaluate=False,
        )
        steps.append(
            _equivalent_display_step(
                rule="Multiply both sides by the denominator",
                semantic_before=current,
                semantic_after=multiplied,
                display_before=current,
                display_after=multiplied_display,
                explanation=(
                    "Multiply each side by the common denominator. The domain restrictions "
                    "make this multiplier nonzero."
                ),
                variable=variable,
                excluded_values=excluded_values,
                introduced_constraints=restrictions.displayed,
                verification_detail=(
                    "Multiplying both sides by the same nonzero expression preserves the "
                    "solution set."
                ),
            )
        )
        current = multiplied
        cancellation_display = BackendIdentity(
            left=_cancelled_side(equation.lhs, denominator),
            right=_cancelled_side(equation.rhs, denominator),
        )
        steps.append(
            _equivalent_display_step(
                rule="Cancel the common factors",
                semantic_before=multiplied,
                semantic_after=cleared,
                display_before=cancellation_display,
                display_after=cleared,
                explanation=(
                    "Cancel each denominator with the matching nonzero factor introduced on "
                    "the same side."
                ),
                variable=variable,
                excluded_values=excluded_values,
                verification_detail=("Canceling equal nonzero factors preserves the solution set."),
            )
        )
        current = cleared
        current = _append_if_changed(
            steps,
            rule="Expand and collect like terms",
            before=current,
            after=normalized,
            explanation="Expand the products, move every term to one side, and combine terms.",
            variable=variable,
            excluded_values=excluded_values,
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
    elif degree == _CUBIC_DEGREE:
        detail = _cubic_steps(
            normalized if steps else equation,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    else:
        detail = _generic_polynomial_steps(
            normalized if steps else equation,
            variable,
            polynomial,
            roots,
            excluded_values,
        )
    steps.extend(detail)
    return tuple(steps)
