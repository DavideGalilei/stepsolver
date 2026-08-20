"""Human-readable derivations for polynomial equations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor, isqrt
from typing import TypeGuard, cast

import sympy as sp

from stepsolver.derivation.model import (
    BackendApproximateSolutions,
    BackendCardanoSolution,
    BackendCrossedOut,
    BackendDerivationStep,
    BackendDifference,
    BackendExpression,
    BackendGrouped,
    BackendIdentity,
    BackendIntroducedOperation,
    BackendIntroducedProduct,
    BackendIntroducedQuotient,
    BackendMathNote,
    BackendNewtonIterations,
    BackendNewtonRule,
    BackendNotEqual,
    BackendProduct,
    BackendQuadraticSolutions,
    BackendQuotient,
    BackendStepConstraint,
    BackendSum,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import is_real_expression

_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2
_CUBIC_DEGREE = 3
_CONSTANT_DEGREE = 0
_ROOT_VERIFICATION_DIGITS = 12
_DISPLAY_DIGITS = 7
_NEWTON_ITERATION_COUNT = 3
_MAX_RATIONAL_ROOT_CANDIDATES = 12


@dataclass(frozen=True, slots=True, kw_only=True)
class _DomainRestrictions:
    denominators: tuple[sp.Basic, ...]
    excluded_values: tuple[sp.Basic, ...]
    displayed: tuple[BackendStepConstraint, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _PreparedPolynomialEquation:
    current: sp.Equality
    expanded: sp.Basic
    restrictions: _DomainRestrictions
    cleared_denominators: bool


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


def _positive_divisors(value: int) -> tuple[int, ...]:
    small: list[int] = []
    large: list[int] = []
    for candidate in range(1, isqrt(value) + 1):
        if value % candidate != 0:
            continue
        small.append(candidate)
        paired = value // candidate
        if paired != candidate:
            large.append(paired)
    return (*small, *reversed(large))


def _rational_sort_key(value: sp.Rational) -> float:
    return float(str(value))


def _rational_root_candidates(
    polynomial: sp.Poly,
    variable: sp.Symbol,
) -> tuple[sp.Rational, ...]:
    leading_coefficient = polynomial.coeff_monomial(variable ** polynomial.degree())
    constant_coefficient = polynomial.coeff_monomial(1)
    if not isinstance(leading_coefficient, sp.Integer) or not isinstance(
        constant_coefficient, sp.Integer
    ):
        return ()
    leading = abs(int(str(leading_coefficient)))
    constant = abs(int(str(constant_coefficient)))
    if leading == 0 or constant == 0:
        return ()
    candidates: set[sp.Rational] = {
        sp.Rational(sign * numerator, denominator)
        for numerator in _positive_divisors(constant)
        for denominator in _positive_divisors(leading)
        for sign in (-1, 1)
    }
    return tuple(sorted(candidates, key=_rational_sort_key))


def _newton_values(
    expression: sp.Basic,
    variable: sp.Symbol,
    initial: sp.Basic,
) -> tuple[sp.Basic, ...]:
    derivative = sp.diff(expression, variable)
    current = sp.N(initial, _DISPLAY_DIGITS)
    values: list[sp.Basic] = [current]
    for _index in range(_NEWTON_ITERATION_COUNT):
        slope = sp.N(derivative.subs(variable, current), _DISPLAY_DIGITS + 2)
        if slope == sp.Integer(0):
            break
        current = sp.N(
            current - expression.subs(variable, current) / slope,
            _DISPLAY_DIGITS,
        )
        values.append(current)
    return tuple(values)


def _domain_restrictions(
    denominator: sp.Basic,
    supplied_denominators: tuple[sp.Basic, ...],
    variable: sp.Symbol,
) -> _DomainRestrictions:
    candidates: tuple[sp.Basic, ...] = supplied_denominators
    if not candidates:
        candidates = (denominator,)
    denominators: list[sp.Basic] = []
    for candidate in candidates:
        _constant_part, variable_part = candidate.as_independent(
            variable,
            as_Add=False,
        )
        variable_denominator = sp.factor(variable_part)
        if variable_denominator != sp.Integer(1):
            denominators.append(variable_denominator)
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


def _common_denominator(
    denominator: sp.Basic,
    supplied_denominators: tuple[sp.Basic, ...],
) -> sp.Basic:
    """Find the least common multiplier from the equation's original fractions."""
    common: sp.Basic = sp.Integer(1)
    for candidate in (*supplied_denominators, denominator):
        common = sp.lcm(common, candidate)
    return sp.factor(common)


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


def _prepare_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    domain_denominators: tuple[sp.Basic, ...],
    steps: list[BackendDerivationStep],
) -> _PreparedPolynomialEquation:
    """Clear original fractions and retain the equation used by later human steps."""
    difference = sp.together(equation.lhs - equation.rhs)
    numerator, denominator = sp.fraction(difference)
    clearing_denominator = _common_denominator(denominator, domain_denominators)
    restrictions = _domain_restrictions(denominator, domain_denominators, variable)
    if clearing_denominator == sp.Integer(1):
        return _PreparedPolynomialEquation(
            current=equation,
            expanded=sp.expand(numerator),
            restrictions=restrictions,
            cleared_denominators=False,
        )

    multiplied = sp.Eq(
        _multiplied_side(equation.lhs, clearing_denominator),
        _multiplied_side(equation.rhs, clearing_denominator),
        evaluate=False,
    )
    multiplied_display = BackendIdentity(
        left=BackendIntroducedProduct(
            multiplier=clearing_denominator,
            expression=equation.lhs,
        ),
        right=BackendIntroducedProduct(
            multiplier=clearing_denominator,
            expression=equation.rhs,
        ),
    )
    cleared = sp.Eq(
        _cleared_side(equation.lhs, clearing_denominator),
        _cleared_side(equation.rhs, clearing_denominator),
        evaluate=False,
    )
    steps.append(
        _equivalent_display_step(
            rule="Multiply both sides by the denominator",
            semantic_before=equation,
            semantic_after=multiplied,
            display_before=equation,
            display_after=multiplied_display,
            explanation=(
                "Multiply each side by the least common denominator so every fraction "
                "can be cleared."
            ),
            variable=variable,
            excluded_values=restrictions.excluded_values,
            introduced_constraints=restrictions.displayed,
            verification_detail=(
                "Multiplying both sides by the same nonzero expression preserves the solution set."
            ),
        )
    )
    cancellation_display = BackendIdentity(
        left=_cancelled_side(equation.lhs, clearing_denominator),
        right=_cancelled_side(equation.rhs, clearing_denominator),
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
            excluded_values=restrictions.excluded_values,
            verification_detail="Canceling equal nonzero factors preserves the solution set.",
        )
    )
    expanded = sp.expand(
        numerator if clearing_denominator.has(variable) else cleared.lhs - cleared.rhs
    )
    return _PreparedPolynomialEquation(
        current=cleared,
        expanded=expanded,
        restrictions=restrictions,
        cleared_denominators=True,
    )


def _operation_expression(expression: sp.Basic, term: sp.Basic) -> BackendExpression:
    """Display adding the opposite of a negative term or subtracting a positive term."""
    if term.could_extract_minus_sign():
        return BackendIntroducedOperation(
            expression=expression,
            operand=-term,
            operation="add",
        )
    return BackendIntroducedOperation(
        expression=expression,
        operand=term,
        operation="subtract",
    )


def _plain_operation_expression(expression: sp.Basic, term: sp.Basic) -> BackendExpression:
    if term.could_extract_minus_sign():
        return BackendSum(terms=(expression, -term))
    return BackendDifference(left=expression, right=term)


def _with_constant(expression: BackendExpression, constant: sp.Basic) -> BackendExpression:
    if constant == sp.Integer(0):
        return expression
    if constant.could_extract_minus_sign():
        return BackendDifference(left=expression, right=-constant)
    return BackendSum(terms=(expression, constant))


def _grouped_variable_display(
    equation: sp.Equality,
    removed_term: sp.Basic,
    variable: sp.Symbol,
    *,
    variable_on_left: bool,
) -> BackendIdentity:
    def grouped_side(expression: sp.Basic, *, canceled: bool) -> BackendExpression:
        polynomial = sp.Poly(expression, variable)
        original_term = polynomial.coeff_monomial(variable) * variable
        constant = polynomial.coeff_monomial(1)
        difference = BackendDifference(
            left=BackendCrossedOut(expression=original_term) if canceled else original_term,
            right=BackendCrossedOut(expression=removed_term) if canceled else removed_term,
        )
        return _with_constant(BackendGrouped(expression=difference), constant)

    return BackendIdentity(
        left=grouped_side(equation.lhs, canceled=not variable_on_left),
        right=grouped_side(equation.rhs, canceled=variable_on_left),
    )


def _constant_cancellation_display(
    equation: sp.Equality,
    constant: sp.Basic,
    variable: sp.Symbol,
    *,
    variable_on_left: bool,
) -> BackendIdentity:
    magnitude = -constant if constant.could_extract_minus_sign() else constant

    def variable_side(expression: sp.Basic) -> BackendExpression:
        coefficient = sp.Poly(expression, variable).coeff_monomial(variable)
        variable_term = coefficient * variable
        first = BackendCrossedOut(expression=magnitude)
        second = BackendCrossedOut(expression=magnitude)
        canceled_constants: BackendExpression
        if constant.could_extract_minus_sign():
            canceled_constants = BackendSum(
                terms=(BackendDifference(left=variable_term, right=first), second)
            )
        else:
            canceled_constants = BackendDifference(
                left=BackendSum(terms=(variable_term, first)),
                right=second,
            )
        return canceled_constants

    other_side = equation.rhs if variable_on_left else equation.lhs
    return BackendIdentity(
        left=(
            variable_side(equation.lhs)
            if variable_on_left
            else _plain_operation_expression(other_side, constant)
        ),
        right=(
            _plain_operation_expression(other_side, constant)
            if variable_on_left
            else variable_side(equation.rhs)
        ),
    )


def _division_cancellation_display(
    equation: sp.Equality,
    divisor: sp.Basic,
    variable: sp.Symbol,
    *,
    variable_on_left: bool,
) -> BackendIdentity:
    canceled_variable = BackendQuotient(
        numerator=BackendProduct(factors=(BackendCrossedOut(expression=divisor), variable)),
        denominator=BackendCrossedOut(expression=divisor),
    )
    numeric_side = equation.rhs if variable_on_left else equation.lhs
    numeric_quotient = BackendQuotient(numerator=numeric_side, denominator=divisor)
    return BackendIdentity(
        left=canceled_variable if variable_on_left else numeric_quotient,
        right=numeric_quotient if variable_on_left else canceled_variable,
    )


def _move_term_from_both_sides(
    steps: list[BackendDerivationStep],
    *,
    equation: sp.Equality,
    term: sp.Basic,
    term_name: str,
    variable: sp.Symbol,
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[sp.Equality, BackendIdentity]:
    """Show one balanced add/subtract operation and return its simplified meaning."""
    operation = "Add" if term.could_extract_minus_sign() else "Subtract"
    preposition = "to" if operation == "Add" else "from"
    displayed_term = -term if term.could_extract_minus_sign() else term
    operation_display = BackendIdentity(
        left=_operation_expression(equation.lhs, term),
        right=_operation_expression(equation.rhs, term),
    )
    simplified = sp.Eq(
        sp.expand(equation.lhs - term),
        sp.expand(equation.rhs - term),
        evaluate=False,
    )
    steps.append(
        _equivalent_display_step(
            rule=f"{operation} the {term_name} {preposition} both sides",
            semantic_before=equation,
            semantic_after=simplified,
            display_before=equation,
            display_after=operation_display,
            explanation=f"{operation} {displayed_term} on each side to preserve equality.",
            variable=variable,
            excluded_values=excluded_values,
        )
    )
    return simplified, operation_display


def _linear_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    expanded_equation = sp.Eq(sp.expand(equation.lhs), sp.expand(equation.rhs), evaluate=False)
    current: sp.Equality = equation
    if str(current) != str(expanded_equation):
        current = cast(
            "sp.Equality",
            _append_if_changed(
                steps,
                rule="Expand both sides",
                before=current,
                after=expanded_equation,
                explanation="Expand every remaining product before collecting terms.",
                variable=variable,
                excluded_values=excluded_values,
            ),
        )

    left_polynomial = sp.Poly(current.lhs, variable)
    right_polynomial = sp.Poly(current.rhs, variable)
    net_coefficient = polynomial.coeff_monomial(variable)
    variable_on_left = net_coefficient.is_negative is not True
    variable_term = (
        right_polynomial.coeff_monomial(variable) * variable
        if variable_on_left
        else left_polynomial.coeff_monomial(variable) * variable
    )
    if variable_term != sp.Integer(0):
        before_move = current
        current, operation_display = _move_term_from_both_sides(
            steps,
            equation=current,
            term=variable_term,
            term_name="variable term",
            variable=variable,
            excluded_values=excluded_values,
        )
        grouped_display = _grouped_variable_display(
            before_move,
            variable_term,
            variable,
            variable_on_left=variable_on_left,
        )
        steps.extend(
            (
                _equivalent_display_step(
                    rule="Group like terms",
                    semantic_before=current,
                    semantic_after=current,
                    display_before=operation_display,
                    display_after=grouped_display,
                    explanation=(
                        "Place the variable terms together and mark the opposite pair that cancels."
                    ),
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                _equivalent_display_step(
                    rule="Combine the variable terms",
                    semantic_before=current,
                    semantic_after=current,
                    display_before=grouped_display,
                    display_after=current,
                    explanation="Combine the grouped coefficients and remove the canceled pair.",
                    variable=variable,
                    excluded_values=excluded_values,
                ),
            )
        )

    variable_side = current.lhs if variable_on_left else current.rhs
    variable_side_constant = sp.Poly(variable_side, variable).coeff_monomial(1)
    if variable_side_constant != sp.Integer(0):
        before_move = current
        current, operation_display = _move_term_from_both_sides(
            steps,
            equation=current,
            term=variable_side_constant,
            term_name="constant",
            variable=variable,
            excluded_values=excluded_values,
        )
        cancellation_display = _constant_cancellation_display(
            before_move,
            variable_side_constant,
            variable,
            variable_on_left=variable_on_left,
        )
        steps.extend(
            (
                _equivalent_display_step(
                    rule="Cancel the opposite constants",
                    semantic_before=current,
                    semantic_after=current,
                    display_before=operation_display,
                    display_after=cancellation_display,
                    explanation="The opposite constants on the variable side add to zero.",
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                _equivalent_display_step(
                    rule="Simplify both sides",
                    semantic_before=current,
                    semantic_after=current,
                    display_before=cancellation_display,
                    display_after=current,
                    explanation="Remove the canceled constants and finish the arithmetic.",
                    variable=variable,
                    excluded_values=excluded_values,
                ),
            )
        )

    variable_side = current.lhs if variable_on_left else current.rhs
    divisor = sp.Poly(variable_side, variable).coeff_monomial(variable)
    constant = polynomial.coeff_monomial(1)
    candidate = sp.simplify(-constant / net_coefficient)
    simplified_division = (
        sp.Eq(variable, candidate, evaluate=False)
        if variable_on_left
        else sp.Eq(candidate, variable, evaluate=False)
    )
    if divisor != sp.Integer(1):
        division_display = BackendIdentity(
            left=BackendIntroducedQuotient(
                numerator=current.lhs,
                denominator=divisor,
            ),
            right=BackendIntroducedQuotient(
                numerator=current.rhs,
                denominator=divisor,
            ),
        )
        cancellation_display = _division_cancellation_display(
            current,
            divisor,
            variable,
            variable_on_left=variable_on_left,
        )
        steps.extend(
            (
                _equivalent_display_step(
                    rule="Divide both sides by the coefficient",
                    semantic_before=current,
                    semantic_after=simplified_division,
                    display_before=current,
                    display_after=division_display,
                    explanation=("Divide each side by the coefficient attached to the variable."),
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                _equivalent_display_step(
                    rule="Cancel the common coefficient",
                    semantic_before=simplified_division,
                    semantic_after=simplified_division,
                    display_before=division_display,
                    display_after=cancellation_display,
                    explanation=(
                        "Cancel the matching nonzero coefficient in the variable quotient."
                    ),
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                _equivalent_display_step(
                    rule="Simplify the quotients",
                    semantic_before=simplified_division,
                    semantic_after=simplified_division,
                    display_before=cancellation_display,
                    display_after=simplified_division,
                    explanation=("Remove the canceled factor and simplify the numerical quotient."),
                    variable=variable,
                    excluded_values=excluded_values,
                ),
            )
        )
        current = simplified_division
    if not variable_on_left:
        current = cast(
            "sp.Equality",
            _append_if_changed(
                steps,
                rule="Write the variable on the left",
                before=current,
                after=sp.Eq(variable, candidate, evaluate=False),
                explanation="Reverse the equality so the solved variable appears first.",
                variable=variable,
                excluded_values=excluded_values,
            ),
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
    expression = polynomial.as_expr()
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
    shift = sp.simplify(-coefficient_b / (3 * coefficient_a))
    first_radicand = sp.simplify(-depressed_constant / 2 + sp.sqrt(discriminant))
    second_radicand = sp.simplify(-depressed_constant / 2 - sp.sqrt(discriminant))
    cardano_solution = BackendCardanoSolution(
        variable=variable,
        shift=shift,
        first_radicand=first_radicand,
        second_radicand=second_radicand,
    )
    candidates = _rational_root_candidates(polynomial, variable)
    function = sp.Function("f")
    candidate_checks = tuple(
        sp.Eq(
            function(candidate),
            sp.simplify(expression.subs(variable, candidate)),
            evaluate=False,
        )
        for candidate in candidates
    )
    numerical_root = sp.N(roots[0], _DISPLAY_DIGITS)
    numerical_root_float = float(str(numerical_root))
    lower = sp.Integer(floor(numerical_root_float))
    upper = sp.Integer(ceil(numerical_root_float))
    bracket_checks = tuple(
        sp.Eq(
            function(point),
            sp.simplify(expression.subs(variable, point)),
            evaluate=False,
        )
        for point in (lower, upper)
    )
    derivative = sp.diff(expression, variable)
    initial = lower if derivative.subs(variable, lower) != sp.Integer(0) else upper
    notes: list[BackendMathNote] = []
    if candidate_checks and len(candidate_checks) <= _MAX_RATIONAL_ROOT_CANDIDATES:
        notes.append(
            BackendMathNote(
                label="Rational-root test",
                expression=candidate_checks,
            )
        )
    notes.extend(
        (
            BackendMathNote(label="Bracket the root", expression=bracket_checks),
            BackendMathNote(label="Newton iteration", expression=BackendNewtonRule()),
            BackendMathNote(
                label="Successive estimates",
                expression=BackendNewtonIterations(
                    variable=variable,
                    values=_newton_values(expression, variable, initial),
                ),
            ),
            BackendMathNote(label="Exact form (optional)", expression=cardano_solution),
        )
    )
    return (
        BackendDerivationStep(
            rule="Approximate the real root",
            before=equation,
            after=BackendApproximateSolutions(
                variable=variable,
                roots=(numerical_root,),
            ),
            explanation=(
                "None of the rational-root candidates is a root, so grouping does not give "
                "a rational factorization. Bracket the one real root, then refine it with "
                "Newton's method."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Substitution gives a residual consistent with the displayed precision; the "
                "exact Cardano form is retained as an optional note."
            ),
            notes=tuple(notes),
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
    prepared = _prepare_polynomial_equation(
        equation,
        variable,
        domain_denominators,
        steps,
    )
    restrictions = prepared.restrictions
    excluded_values = restrictions.excluded_values
    current: EquationBackendExpression = prepared.current
    expanded = prepared.expanded
    normalized = sp.Eq(expanded, 0, evaluate=False)
    polynomial = sp.Poly(expanded, variable)
    degree = polynomial.degree()
    if prepared.cleared_denominators and degree != _LINEAR_DEGREE:
        current = _append_if_changed(
            steps,
            rule="Expand and collect like terms",
            before=current,
            after=normalized,
            explanation="Expand the products, move every term to one side, and combine terms.",
            variable=variable,
            excluded_values=excluded_values,
        )
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
            prepared.current,
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
