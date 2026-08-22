"""Human-readable transformations for linear equations."""

from __future__ import annotations

from typing import cast

import sympy as sp

from stepsolver.derivation.equations_support import append_if_changed, equivalent_display_step
from stepsolver.derivation.model import (
    BackendCrossedOut,
    BackendDerivationStep,
    BackendDifference,
    BackendExpression,
    BackendGrouped,
    BackendIdentity,
    BackendIntroducedOperation,
    BackendIntroducedQuotient,
    BackendProduct,
    BackendQuotient,
    BackendSum,
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
        equivalent_display_step(
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


def derive_linear_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Isolate one variable through explicit balance-preserving operations."""
    steps: list[BackendDerivationStep] = []
    expanded_equation = sp.Eq(sp.expand(equation.lhs), sp.expand(equation.rhs), evaluate=False)
    current: sp.Equality = equation
    if str(current) != str(expanded_equation):
        current = cast(
            "sp.Equality",
            append_if_changed(
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
                equivalent_display_step(
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
                equivalent_display_step(
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
                equivalent_display_step(
                    rule="Cancel the opposite constants",
                    semantic_before=current,
                    semantic_after=current,
                    display_before=operation_display,
                    display_after=cancellation_display,
                    explanation="The opposite constants on the variable side add to zero.",
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                equivalent_display_step(
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
                equivalent_display_step(
                    rule="Divide both sides by the coefficient",
                    semantic_before=current,
                    semantic_after=simplified_division,
                    display_before=current,
                    display_after=division_display,
                    explanation=("Divide each side by the coefficient attached to the variable."),
                    variable=variable,
                    excluded_values=excluded_values,
                ),
                equivalent_display_step(
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
                equivalent_display_step(
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
            append_if_changed(
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
        append_if_changed(
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
