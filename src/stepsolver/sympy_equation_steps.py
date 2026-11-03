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
    OpaqueExpression,
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
        parsed = self._inequality_expression(query)
        if parsed is None:
            return ()
        original, variable, difference = parsed
        operator = original.operator.value
        chart = self._sign_chart(difference, variable)
        if chart is None:
            return ()
        zeros, poles = chart
        boundaries = sorted(
            set(zeros) | set(poles),
            key=lambda root: float(str(sp.N(root, 15))),
        )
        backend_set = self._backend_solution_set(backend_value)
        if backend_set is None:
            return ()
        reduced = Relation(
            operator=original.operator,
            left=self._converter.from_sympy(difference),
            right=self._integer(0),
        )
        move_step = self._step(
            rule="Move every term to one side",
            before=query_expression(query),
            after=reduced,
            explanation=(
                "Subtract the right side from both sides so the sign is compared with zero."
            ),
            method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            detail="Subtracting the same quantity from both sides keeps the solution set.",
        )
        if not boundaries:
            return self._constant_sign_inequality_steps(
                operator,
                difference,
                variable,
                backend_set=backend_set,
                reduced=reduced,
                move_step=move_step,
            )
        tested = self._test_intervals(operator, difference, variable, boundaries)
        solution_set = self._inequality_solution_set(operator, tested, zeros, poles)
        if solution_set is None or solution_set != backend_set:
            return ()
        boundary = SequenceExpression(
            items=tuple(self._converter.from_sympy(root) for root in boundaries)
        )
        return (
            move_step,
            self._step(
                rule="Find the boundary points",
                before=reduced,
                after=boundary,
                explanation=self._boundary_explanation(difference, has_poles=bool(poles)),
                method=VerificationMethod.SUBSTITUTION,
                detail=(
                    "Substituting each zero makes the numerator exactly zero, and each pole "
                    "makes the original denominator zero."
                    if poles
                    else "Substituting each boundary point makes the expression exactly zero."
                ),
                notes=(
                    *(
                        StepNote(
                            label=f"Zero at {root}",
                            expression=self._converter.from_sympy(
                                sp.simplify(difference.subs(variable, root))
                            ),
                        )
                        for root in zeros
                    ),
                    *(
                        StepNote(
                            label=f"Undefined pole at {root}",
                            expression=OpaqueExpression(text="undefined"),
                        )
                        for root in poles
                    ),
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
                        if operator in {"<", ">"}
                        else ", together with the zeros that are not excluded by a pole."
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

    def _inequality_expression(
        self,
        query: Query,
    ) -> tuple[Relation, sp.Symbol, sp.Basic] | None:
        """Parse an inequality and reduce it to one difference from zero."""
        original = query.arguments[0]
        variable = self._converter.to_sympy(query.arguments[1])
        if not isinstance(original, Relation) or not isinstance(variable, sp.Symbol):
            return None
        difference = sp.together(
            self._converter.to_sympy(original.left) - self._converter.to_sympy(original.right)
        )
        if difference.free_symbols != {variable}:
            return None
        return original, variable, difference

    @staticmethod
    def _backend_solution_set(backend_value: sp.Basic) -> sp.Basic | None:
        """Convert the backend relation to a set, treating the universal set as the reals."""
        try:
            backend_set = backend_value.as_set()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None
        universal_set = cast("sp.Basic", getattr(sp, "UniversalSet"))
        if backend_set is universal_set:
            return cast("sp.Basic", getattr(sp, "Reals"))
        return backend_set

    @staticmethod
    def _inequality_solution_set(
        operator: str,
        tested: list[tuple[sp.Interval, bool, sp.Basic, sp.Basic]],
        zeros: tuple[sp.Basic, ...],
        poles: tuple[sp.Basic, ...],
    ) -> sp.Basic | None:
        """Assemble the kept intervals and eligible boundary roots into one set."""
        if not tested:
            return None
        included = [interval for interval, passed, _value, _point in tested if passed]
        included_roots = (
            [root for root in zeros if root not in poles] if operator not in {"<", ">"} else []
        )
        return sp.Union(*included, sp.FiniteSet(*included_roots))

    def _constant_sign_inequality_steps(
        self,
        operator: str,
        difference: sp.Basic,
        variable: sp.Symbol,
        *,
        backend_set: sp.Basic,
        reduced: Relation,
        move_step: SolutionStep,
    ) -> tuple[SolutionStep, ...]:
        """Explain an inequality whose expression never changes sign on the real line."""
        point = sp.Integer(0)
        value = sp.simplify(difference.subs(variable, point))
        holds = self._comparison_holds(operator, value)
        if holds is None:
            return ()
        solution_set = cast("sp.Basic", getattr(sp, "Reals")) if holds else sp.FiniteSet()
        if solution_set != backend_set:
            return ()
        return (
            move_step,
            self._step(
                rule="Check the sign for every real number",
                before=reduced,
                after=OpaqueExpression(text="all real numbers" if holds else "no real solutions"),
                explanation=(
                    "The expression has no zero and no pole, so it cannot change sign. "
                    + (
                        "Testing one real number shows that the inequality holds everywhere."
                        if holds
                        else "Testing one real number shows that the inequality never holds."
                    )
                ),
                method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                detail="The single sign check agrees with the backend solution set.",
                notes=(
                    StepNote(
                        label=f"At {variable} = {point}",
                        expression=self._converter.from_sympy(value),
                    ),
                ),
            ),
        )

    @staticmethod
    def _boundary_explanation(difference: sp.Basic, *, has_poles: bool) -> str:
        if has_poles:
            return (
                "A rational expression can change sign only where its numerator is zero or "
                "its denominator is zero. Poles are excluded from the solution."
            )
        if difference.has(sp.Abs):
            return (
                "An absolute-value expression can change sign only where the whole expression "
                "equals zero, so solve that equation for the boundary points."
            )
        return (
            "A polynomial can change sign only where it equals zero, so solve the "
            "polynomial equal to zero."
        )

    @staticmethod
    def _sign_chart(
        difference: sp.Basic,
        variable: sp.Symbol,
    ) -> tuple[tuple[sp.Basic, ...], tuple[sp.Basic, ...]] | None:
        """Return the real zeros of the numerator and the real poles of the denominator."""
        numerator, denominator = sp.fraction(difference)
        zeros = SympyEquationStepBuilder._real_solutions(numerator, variable)
        if zeros is None:
            return None
        if denominator.has(variable):
            poles = SympyEquationStepBuilder._real_solutions(denominator, variable)
            if poles is None:
                return None
        else:
            poles = ()
        return zeros, poles

    @staticmethod
    def _real_solutions(
        expression: sp.Basic,
        variable: sp.Symbol,
    ) -> tuple[sp.Basic, ...] | None:
        """Solve one univariate expression for its complete finite set of real zeros."""
        if not expression.has(variable):
            return ()
        try:
            raw_solutions = sp.solve(sp.Eq(expression, 0), variable)
        except (NotImplementedError, TypeError, ValueError):
            return None
        if not isinstance(raw_solutions, list):
            return None
        solutions = cast("list[object]", raw_solutions)
        real_solutions: list[sp.Basic] = []
        for candidate in solutions:
            if not isinstance(candidate, sp.Basic):
                return None
            solution = candidate
            if solution.is_real is True:
                real_solutions.append(solution)
            elif solution.is_real is not False:
                return None
        return tuple(real_solutions)

    @staticmethod
    def _comparison_holds(operator: str, value: sp.Basic) -> bool | None:
        comparisons = {
            "<": sp.Lt(value, sp.Integer(0)),
            ">": sp.Gt(value, sp.Integer(0)),
            "<=": sp.Le(value, sp.Integer(0)),
            ">=": sp.Ge(value, sp.Integer(0)),
        }
        comparison = comparisons.get(operator)
        if comparison is None:
            return None
        if comparison is sp.true:
            return True
        if comparison is sp.false:
            return False
        return None

    @staticmethod
    def _test_intervals(
        operator: str,
        difference: sp.Basic,
        variable: sp.Symbol,
        boundaries: list[sp.Basic],
    ) -> list[tuple[sp.Interval, bool, sp.Basic, sp.Basic]]:
        """Evaluate the sign of the expression at one exact point inside each open interval."""
        bounds = [-sp.oo, *boundaries, sp.oo]
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
            }.get(operator)
            if comparison is None:
                return []
            results.append((sp.Interval.open(lower, upper), bool(comparison), value, point))
        return results
