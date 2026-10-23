"""Student-facing derivations for differential equations, recurrences, and inequalities.

The ODE and recurrence builders do not claim to re-derive the solution method.  They state
the classification, present the solution family, and verify the solution independently by
substitution.  The inequality builder performs the full sign-chart method.
"""

from fractions import Fraction
from itertools import pairwise
from typing import cast

import sympy as sp

from stepsolver.ast import (
    Expression,
    Number,
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
)
from stepsolver.results import SolutionStep, StepNote, Verification, VerificationMethod
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_support import query_expression

_EQUATION_ARITY = 2
_CONSTANT_PREFIX = "C"


class SympyEquationStepBuilder:
    """Build verified human steps for ODE, recurrence, and inequality queries."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the shared converter to display backend values as public expressions."""
        self._converter = converter

    def steps(self, query: Query, backend_value: object) -> tuple[SolutionStep, ...]:
        """Return steps for the query, or an empty tuple when none can be verified."""
        if len(query.arguments) != _EQUATION_ARITY or not isinstance(backend_value, sp.Basic):
            return ()
        match query.operation:
            case Operation.DSOLVE:
                return self._ode_steps(query, backend_value)
            case Operation.RSOLVE:
                return self._recurrence_steps(query, backend_value)
            case Operation.SOLVE_INEQUALITY:
                return self._inequality_steps(query, backend_value)
            case _:
                return ()

    @staticmethod
    def _integer(value: int) -> Number:
        return Number(value=Fraction(value))

    @staticmethod
    def _step(
        *,
        rule: str,
        before: Expression,
        after: Expression,
        explanation: str,
        method: VerificationMethod,
        detail: str,
        notes: tuple[StepNote, ...] = (),
    ) -> SolutionStep:
        return SolutionStep(
            rule=rule,
            before=before,
            after=after,
            explanation=explanation,
            verification=Verification(method=method, detail=detail),
            notes=notes,
        )

    @staticmethod
    def _constants(value: sp.Basic) -> tuple[sp.Symbol, ...]:
        names = sorted(
            (symbol for symbol in value.free_symbols if symbol.name.startswith(_CONSTANT_PREFIX)),
            key=lambda symbol: symbol.name,
        )
        return tuple(names)

    def _constants_note(self, constants: tuple[sp.Symbol, ...]) -> StepNote:
        return StepNote(
            label="Arbitrary constants",
            expression=SequenceExpression(
                items=tuple(self._converter.from_sympy(symbol) for symbol in constants)
            ),
        )

    # Differential equations ----------------------------------------------------------

    def _ode_steps(self, query: Query, backend_value: sp.Basic) -> tuple[SolutionStep, ...]:
        equation = self._converter.to_sympy(query.arguments[0])
        function = self._converter.to_sympy(query.arguments[1])
        if not isinstance(equation, sp.Equality) or not isinstance(backend_value, sp.Equality):
            return ()
        verified, residual = sp.checkodesol(equation, backend_value, function)
        if verified is not True:
            return ()
        order = sp.ode_order(equation.lhs - equation.rhs, function)
        constants = self._constants(backend_value)
        hints = sp.classify_ode(equation, function)
        if order < 1 or len(constants) != order or not hints:
            return ()
        solution = self._converter.step_expression(backend_value)
        return (
            self._step(
                rule="Identify the differential equation",
                before=query_expression(query),
                after=query.arguments[0],
                explanation=(
                    f"The equation is an ordinary differential equation of order {order} for the "
                    "unknown function."
                ),
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The highest derivative order was read from the equation.",
                notes=(
                    StepNote(label="Order", expression=self._integer(order)),
                    StepNote(label="Unknown function", expression=query.arguments[1]),
                ),
            ),
            self._step(
                rule="Apply the classified solution method",
                before=query.arguments[0],
                after=solution,
                explanation=(
                    f"The equation is classified as '{hints[0]}', and that method gives a "
                    f"solution family with {order} arbitrary constant(s). The individual "
                    "integration steps of that method are not expanded here; the next step "
                    "checks the family independently."
                ),
                method=VerificationMethod.BACKEND_IDENTITY,
                detail="The symbolic ODE solver returned this family for the classified method.",
                notes=(self._constants_note(constants),),
            ),
            self._step(
                rule="Check the solution by substitution",
                before=solution,
                after=Relation(
                    operator=RelationOperator.EQUAL,
                    left=self._converter.step_expression(residual),
                    right=self._integer(0),
                ),
                explanation=(
                    "Substitute the solution into the differential equation; the difference "
                    "between both sides must simplify to zero for every constant."
                ),
                method=VerificationMethod.SUBSTITUTION,
                detail="Substituting the solution leaves a residual of exactly zero.",
                notes=(self._constants_note(constants),),
            ),
        )

    # Recurrences ---------------------------------------------------------------------

    def _recurrence_steps(self, query: Query, backend_value: sp.Basic) -> tuple[SolutionStep, ...]:
        recurrence = self._converter.to_sympy(query.arguments[0])
        term = self._converter.to_sympy(query.arguments[1])
        if len(term.args) != 1:
            return ()
        index = term.args[0]
        residual_expression = (
            recurrence.lhs - recurrence.rhs if isinstance(recurrence, sp.Equality) else recurrence
        )
        shifts = {
            application.args[0] - index
            for application in residual_expression.atoms(sp.Function)
            if application.func == term.func and len(application.args) == 1
        }
        if not shifts or not all(shift.is_integer for shift in shifts):
            return ()
        largest_shift = cast("sp.Integer", max(shifts, key=lambda shift: int(str(shift))))
        smallest_shift = cast("sp.Integer", min(shifts, key=lambda shift: int(str(shift))))
        order = int(largest_shift) - int(smallest_shift)
        constants = self._constants(backend_value)

        def substitute(argument: sp.Basic) -> sp.Basic:
            return backend_value.subs(index, argument)

        substituted = residual_expression.replace(term.func, substitute)
        if order < 1 or len(constants) != order or sp.simplify(substituted) != sp.Integer(0):
            return ()
        solution = Relation(
            operator=RelationOperator.EQUAL,
            left=query.arguments[1],
            right=self._converter.from_sympy(backend_value),
        )
        return (
            self._step(
                rule="Identify the recurrence",
                before=query_expression(query),
                after=query.arguments[0],
                explanation=(
                    f"The terms of the sequence span {order + 1} consecutive indices, so the "
                    f"recurrence has order {order}."
                ),
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The index offsets of the unknown sequence were read from the equation.",
                notes=(StepNote(label="Order", expression=self._integer(order)),),
            ),
            self._step(
                rule="Write the general solution",
                before=query.arguments[0],
                after=solution,
                explanation=(
                    f"The solution has {order} arbitrary constant(s), fixed by initial values. "
                    "The recurrence solver's internal method is not expanded here; the next step "
                    "checks the formula independently."
                ),
                method=VerificationMethod.BACKEND_IDENTITY,
                detail="The symbolic recurrence solver returned this closed form.",
                notes=(self._constants_note(constants),),
            ),
            self._step(
                rule="Check the solution by substitution",
                before=solution,
                after=Relation(
                    operator=RelationOperator.EQUAL,
                    left=self._integer(0),
                    right=self._integer(0),
                ),
                explanation=(
                    "Replace every term of the sequence in the recurrence by the closed form; "
                    "both sides must agree for every index."
                ),
                method=VerificationMethod.SUBSTITUTION,
                detail="Substituting the closed form leaves a difference of exactly zero.",
                notes=(self._constants_note(constants),),
            ),
        )

    # Inequalities --------------------------------------------------------------------

    def _inequality_steps(self, query: Query, backend_value: sp.Basic) -> tuple[SolutionStep, ...]:
        original = query.arguments[0]
        variable_expression = query.arguments[1]
        inequality = self._converter.to_sympy(original)
        variable = self._converter.to_sympy(variable_expression)
        if (
            not isinstance(original, Relation)
            or not isinstance(variable, sp.Symbol)
            or not isinstance(inequality, sp.Rel)
        ):
            return ()
        difference = sp.expand(inequality.lhs - inequality.rhs)
        if difference.free_symbols != {variable} or not difference.is_polynomial(variable):
            return ()
        roots = sorted(
            set(sp.real_roots(difference)),
            key=lambda root: float(str(sp.N(root, 15))),
        )
        if not roots:
            return ()
        tested = self._test_intervals(inequality, difference, variable, roots)
        included: list[sp.Basic] = [
            interval for interval, passed, _value, _point in tested if passed
        ]
        if inequality.rel_op not in {"<", ">"}:
            included.append(sp.FiniteSet(*roots))
        solution_set = sp.Union(*included)
        try:
            backend_set = backend_value.as_set()
        except (NotImplementedError, TypeError, ValueError):
            return ()
        if backend_set != solution_set:
            return ()
        reduced = Relation(
            operator=original.operator,
            left=self._converter.from_sympy(difference),
            right=self._integer(0),
        )
        boundary = SequenceExpression(
            items=tuple(self._converter.from_sympy(root) for root in roots)
        )
        return (
            self._step(
                rule="Move every term to one side",
                before=query_expression(query),
                after=reduced,
                explanation=(
                    "Subtract the right side from both sides so the sign is compared with zero."
                ),
                method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                detail="Subtracting the same quantity from both sides keeps the solution set.",
            ),
            self._step(
                rule="Find the boundary points",
                before=reduced,
                after=boundary,
                explanation=(
                    "A polynomial can change sign only where it equals zero, so solve the "
                    "polynomial equal to zero."
                ),
                method=VerificationMethod.SUBSTITUTION,
                detail="Substituting each boundary point makes the polynomial exactly zero.",
                notes=tuple(
                    StepNote(
                        label=f"Polynomial at {root}",
                        expression=self._converter.from_sympy(
                            sp.simplify(difference.subs(variable, root))
                        ),
                    )
                    for root in roots
                ),
            ),
            self._step(
                rule="Test one point in each interval",
                before=boundary,
                after=self._converter.step_expression(backend_value),
                explanation=(
                    "Pick a point inside every interval between boundary points and keep the "
                    "intervals where the inequality is true"
                    + (
                        "."
                        if inequality.rel_op in {"<", ">"}
                        else ", together with the boundary points."
                    )
                ),
                method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                detail="The kept intervals form the same set as the backend solution.",
                notes=tuple(
                    StepNote(
                        label=(
                            f"{'Holds' if passed else 'Fails'} on {interval} "
                            f"(test {variable} = {point})"
                        ),
                        expression=self._converter.from_sympy(value),
                    )
                    for interval, passed, value, point in tested
                ),
            ),
        )

    @staticmethod
    def _test_intervals(
        inequality: sp.Rel,
        difference: sp.Basic,
        variable: sp.Symbol,
        roots: list[sp.Basic],
    ) -> list[tuple[sp.Interval, bool, sp.Basic, sp.Basic]]:
        """Evaluate the polynomial at one exact point inside each open interval."""
        bounds = [-sp.oo, *roots, sp.oo]
        results: list[tuple[sp.Interval, bool, sp.Basic, sp.Basic]] = []
        for lower, upper in pairwise(bounds):
            if lower == -sp.oo:
                point = upper - 1
            elif upper == sp.oo:
                point = lower + 1
            else:
                point = (lower + upper) / 2
            value = sp.simplify(difference.subs(variable, point))
            comparison = {
                "<": sp.Lt(value, sp.Integer(0)),
                ">": sp.Gt(value, sp.Integer(0)),
                "<=": sp.Le(value, sp.Integer(0)),
                ">=": sp.Ge(value, sp.Integer(0)),
            }.get(inequality.rel_op)
            if comparison is None:
                return []
            results.append((sp.Interval.open(lower, upper), bool(comparison), value, point))
        return results
