"""SymPy implementation of the StepSolver symbolic backend."""

from fractions import Fraction

import sympy as sp

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Constant,
    ConstantName,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    OpaqueExpression,
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
    SolutionDomain,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.derivation.equations_support import original_denominator_restrictions
from stepsolver.derivation.sums import find_undefined_summation
from stepsolver.errors import BackendError, QueryError
from stepsolver.results import (
    DivergenceKind,
    DivergentResult,
    ExactResult,
    NoSolutionValue,
    NumericResult,
    NumericRootsValue,
    SolutionStep,
    SolveResult,
    StepConstraint,
    StepNote,
    UndefinedResult,
    UniversalSolutionValue,
    UnsolvedResult,
    Verification,
    VerificationMethod,
)
from stepsolver.sympy_contours import ContourExpression, evaluate_rational_circle
from stepsolver.sympy_conversion import SympyConverter, unsupported_function_name
from stepsolver.sympy_equivalence import expressions_equivalent
from stepsolver.sympy_execution import SympyExecutor
from stepsolver.sympy_numeric import solve_numeric
from stepsolver.sympy_rendering import SympyDerivationRenderer
from stepsolver.sympy_steps import SympyStepBuilder, domain_denominators
from stepsolver.sympy_support import (
    contains_unevaluated_operation,
    expect_arity,
    expect_symbol,
    query_expression,
    substitute,
)
from stepsolver.sympy_verification import SympyVerifier

_EQUATION_ARITY = 2
_DEFINITE_INTEGRAL_ARITY = 4
_BOUNDED_OPERATION_ARITY = 4

# Calculus operations may legitimately carry an abstract function such as f(x)
# or the dependent variable y(x) in an ODE.  Their unevaluated backend objects
# are already rejected by the exact-result boundary.  Algebra operations must
# use the documented function registry instead.
_SYMBOLIC_FUNCTION_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.DIFFERENTIATE,
        Operation.INTEGRATE,
        Operation.LIMIT,
        Operation.SERIES,
        Operation.SUM,
        Operation.PRODUCT,
        Operation.DSOLVE,
        Operation.RSOLVE,
    }
)

_INDEFINITE_INTEGRAL_ARGUMENTS = 2


class SympyBackend:
    """Solve StepSolver queries through focused SymPy collaborators."""

    def __init__(self) -> None:
        """Create the converter, executor, renderer, step builder, and verifier."""
        self._converter = SympyConverter()
        self._executor = SympyExecutor(self._converter)
        self._renderer = SympyDerivationRenderer(self._converter)
        self._steps = SympyStepBuilder(self._converter, self._renderer)
        self._verifier = SympyVerifier(self._converter)

    def solve(self, query: Query) -> SolveResult:
        """Solve a validated query and translate the result to StepSolver values."""
        self._validate_query(query)
        try:
            return self._solve(query)
        except ZeroDivisionError:
            return UndefinedResult(
                query=query,
                reason="This expression is undefined because a denominator is zero.",
                steps=(),
            )
        except (BackendError, NotImplementedError, QueryError, TypeError, ValueError) as error:
            return UnsolvedResult(query=query, reason=str(error), steps=())

    def _solve(self, query: Query) -> SolveResult:
        precheck = self._precheck_result(query)
        if precheck is not None:
            return precheck
        if query.operation is Operation.NSOLVE:
            return self._solve_numeric(query)
        try:
            backend_value = self._executor.execute(query)
        except ValueError:
            nonexistent_limit = self._nonexistent_limit_result(query)
            if nonexistent_limit is None:
                raise
            return nonexistent_limit
        detailed_steps = self._steps.detailed_steps(query, backend_value)
        divergence = self._divergence_kind(query, backend_value, detailed_steps)
        if divergence is not None:
            reasons = self._divergence_reasons(query.operation)
            return DivergentResult(
                query=query,
                kind=divergence,
                reason=reasons[divergence],
                steps=detailed_steps,
            )
        non_exact_reason = self._non_exact_reason(backend_value)
        if non_exact_reason is not None:
            return UnsolvedResult(
                query=query,
                reason=non_exact_reason,
                steps=detailed_steps,
            )
        value = (
            NoSolutionValue()
            if query.operation is Operation.SOLVE and backend_value == []
            else self._converter.to_value(backend_value)
        )
        if detailed_steps:
            return ExactResult(query=query, value=value, steps=detailed_steps)
        generic_step = self._generic_exact_step(query, backend_value)
        if generic_step is not None:
            return ExactResult(query=query, value=value, steps=(generic_step,))
        return UnsolvedResult(
            query=query,
            reason=(
                "The symbolic backend computed a result, but it could not be independently "
                "verified for this input."
            ),
            steps=(),
        )

    def _generic_exact_step(
        self,
        query: Query,
        backend_value: object,
    ) -> SolutionStep | None:
        """Return one honest verified step when no detailed template matched."""
        rule, explanation, before = self._generic_step_description(query)
        try:
            verification = self._verifier.verify_result(query, backend_value)
        except (BackendError, QueryError, TypeError, ValueError):
            return None
        constraints: tuple[StepConstraint, ...] = ()
        if (
            query.operation is Operation.INTEGRATE
            and len(query.arguments) == _INDEFINITE_INTEGRAL_ARGUMENTS
        ):
            constraints = self._steps.integration_domain_constraints(query.arguments[0])
        detail = verification.detail
        if constraints:
            detail = (
                f"{detail} The stated exclusions keep every original denominator nonzero, so the "
                "antiderivative is valid on each remaining real interval."
            )
        return SolutionStep(
            rule=rule,
            before=before,
            after=self._converter.step_expression(backend_value),
            explanation=explanation,
            verification=Verification(method=verification.method, detail=detail),
            introduced_constraints=constraints,
        )

    @staticmethod
    def _generic_step_description(query: Query) -> tuple[str, str, Expression]:
        """Choose an operation-specific label for a verified but unexpanded result."""
        operation = query.operation
        if operation is Operation.INTEGRATE:
            if len(query.arguments) == _INDEFINITE_INTEGRAL_ARGUMENTS:
                return (
                    "Verify the antiderivative by differentiation",
                    (
                        "The symbolic backend found an antiderivative. Differentiating it recovers "
                        "the integrand on the intervals allowed by the stated domain restrictions; "
                        "the individual integration technique is not expanded for this input."
                    ),
                    query_expression(query),
                )
            return (
                "Evaluate the definite integral exactly",
                (
                    "The symbolic backend evaluated the definite integral over the stated bounds. "
                    "The reported value reflects the endpoint and convergence checks for this "
                    "integral."
                ),
                query_expression(query),
            )
        if operation is Operation.SOLVE_INEQUALITY:
            return (
                "State the verified solution set",
                (
                    "The inequality was solved exactly over the real line. The displayed set is "
                    "the complete solution, and its endpoints follow the strict or non-strict "
                    "comparison."
                ),
                query.arguments[0],
            )
        if operation is Operation.DSOLVE:
            return (
                "State the verified solution family",
                (
                    "The differential-equation solver returned this solution family. "
                    "Substituting it into the equation leaves a residual of zero for every "
                    "arbitrary constant; the internal integration steps are not expanded for "
                    "this input."
                ),
                query.arguments[0],
            )
        if operation is Operation.RSOLVE:
            return (
                "State the verified recurrence solution",
                (
                    "The recurrence solver returned this closed form. Substituting it into the "
                    "recurrence leaves a difference of zero for every index; the internal "
                    "generating steps are not expanded for this input."
                ),
                query.arguments[0],
            )
        if operation in {
            Operation.MATRIX,
            Operation.DETERMINANT,
            Operation.INVERSE,
            Operation.RANK,
            Operation.RREF,
            Operation.EIGENVALUES,
        }:
            return (
                "State the verified matrix result",
                (
                    "The matrix operation was evaluated exactly and the displayed value is the "
                    "verified result."
                ),
                query_expression(query),
            )
        return (
            "Verify the exact result",
            (
                "The symbolic backend computed this exact value and the result was checked before "
                "display."
            ),
            query_expression(query),
        )

    def _solve_numeric(self, query: Query) -> SolveResult:
        """Solve a real numerical root problem with a verified Newton derivation."""
        expect_arity(query, 2, 3)
        equation_expression, variable_expression, *remaining = query.arguments
        variable = expect_symbol(variable_expression, role="numerical solve variable")
        backend_variable = self._converter.to_sympy(variable)
        if not isinstance(backend_variable, sp.Symbol):
            raise QueryError("nsolve requires a single variable symbol")
        original: sp.Basic
        residual: sp.Basic
        if isinstance(equation_expression, Relation):
            if equation_expression.operator is not RelationOperator.EQUAL:
                raise QueryError("nsolve requires an equation or a residual expression")
            backend_equation = self._converter.to_sympy(equation_expression)
            if not isinstance(backend_equation, sp.Equality):
                raise QueryError("nsolve could not interpret the supplied equation")
            original = backend_equation
            residual = backend_equation.lhs - backend_equation.rhs
            from_equation = True
        else:
            original = self._converter.to_sympy(equation_expression)
            residual = original
            from_equation = False
        guess = self._converter.to_sympy(remaining[0]) if remaining else None
        numeric = solve_numeric(
            original=original,
            residual=residual,
            variable=backend_variable,
            guess=guess,
            from_equation=from_equation,
        )
        steps = tuple(self._renderer.solution_step(step) for step in numeric.steps)
        if not numeric.roots:
            return ExactResult(query=query, value=NoSolutionValue(), steps=steps)
        roots = tuple(ApproximateNumber(text=root.display) for root in numeric.roots)
        return NumericResult(
            query=query,
            value=NumericRootsValue(roots=roots, tolerance=numeric.tolerance),
            steps=steps,
            method=numeric.method,
            residuals=tuple(float(str(root.residual)) for root in numeric.roots),
        )

    def _validate_query(self, query: Query) -> None:
        """Reject structurally invalid queries before any backend execution.

        These failures describe the request itself rather than a mathematical
        obstruction, so they surface as typed query errors instead of an
        ``UnsolvedResult``. That keeps an unanswerable request distinguishable
        from an honest symbolic decline.
        """
        unsupported_reason = self._unsupported_function_reason(query)
        if unsupported_reason is not None:
            raise QueryError(unsupported_reason)
        missing_target = self._missing_target_variable_reason(query)
        if missing_target is not None:
            raise QueryError(missing_target)

    def _missing_target_variable_reason(self, query: Query) -> str | None:
        """Detect a solve target that the equation never mentions."""
        if query.operation is not Operation.SOLVE or len(query.arguments) != _EQUATION_ARITY:
            return None
        equation_expression, variable_expression = query.arguments
        if not isinstance(equation_expression, Relation) or not isinstance(
            variable_expression, Symbol
        ):
            return None
        equation = self._converter.to_sympy(equation_expression)
        variable = self._converter.to_sympy(variable_expression)
        if not isinstance(equation, sp.Rel) or not isinstance(variable, sp.Symbol):
            return None
        symbols = equation.free_symbols
        if symbols and variable not in symbols:
            return f"Target variable '{variable_expression.name}' does not appear in equation"
        return None

    def _precheck_result(self, query: Query) -> SolveResult | None:
        """Handle undefined cases before backend execution."""
        if query.operation is Operation.CONTOUR_INTEGRATE:
            return self._solve_contour(query)
        undefined_sum = self._undefined_sum_result(query)
        if undefined_sum is not None:
            return undefined_sum
        undefined_division = self._undefined_division_result(query)
        if undefined_division is not None:
            return undefined_division
        identity_result = self._identity_equation_result(query)
        if identity_result is not None:
            return identity_result
        identity_reason = self._identity_equation_reason(query)
        if identity_reason is not None:
            return UnsolvedResult(query=query, reason=identity_reason, steps=())
        return None

    @staticmethod
    def _unsupported_function_reason(query: Query) -> str | None:
        """Reject unchecked function names before they can be echoed as an answer."""
        allow_abstract = query.operation in _SYMBOLIC_FUNCTION_OPERATIONS
        allowed_names: frozenset[str] = frozenset()
        if (
            query.operation in {Operation.DSOLVE, Operation.RSOLVE}
            and len(query.arguments) == _EQUATION_ARITY
            and isinstance(query.arguments[1], FunctionCall)
        ):
            allowed_names = frozenset({str(query.arguments[1].name)})
        for argument in query.arguments:
            name = unsupported_function_name(
                argument,
                allow_abstract=allow_abstract,
                allowed_names=allowed_names,
            )
            if name is not None:
                return (
                    f"Unsupported function: {name}. StepSolver only evaluates documented functions."
                )
        return None

    def _undefined_division_result(self, query: Query) -> UndefinedResult | None:
        """Return the typed undefined result for an identically zero denominator."""
        for argument in query.arguments:
            denominator = self._zero_denominator(argument)
            if denominator is not None:
                return UndefinedResult(
                    query=query,
                    reason="This expression is undefined because a denominator is zero.",
                    steps=self._undefined_denominator_steps(query, denominator),
                )
        return None

    def _zero_denominator(self, expression: Expression) -> Expression | None:
        """Locate the first denominator that simplifies to zero."""
        if isinstance(expression, BinaryExpression):
            return self._binary_zero_denominator(expression)
        if isinstance(expression, UnaryExpression):
            return self._zero_denominator(expression.operand)
        if isinstance(expression, FunctionCall):
            return self._first_zero_denominator(expression.arguments)
        if isinstance(expression, Relation):
            return self._first_zero_denominator((expression.left, expression.right))
        if isinstance(expression, SequenceExpression):
            return self._first_zero_denominator(expression.items)
        return None

    def _binary_zero_denominator(self, expression: BinaryExpression) -> Expression | None:
        """Check the two operands of a binary expression for a zero denominator."""
        if expression.operator is BinaryOperator.DIVIDE and self._simplifies_to_zero(
            expression.right
        ):
            return expression.right
        if (
            expression.operator is BinaryOperator.POWER
            and self._simplifies_to_negative(expression.right)
            and self._simplifies_to_zero(expression.left)
        ):
            return expression.left
        return self._first_zero_denominator((expression.left, expression.right))

    def _first_zero_denominator(self, expressions: tuple[Expression, ...]) -> Expression | None:
        """Return the first nested denominator that simplifies to zero."""
        for expression in expressions:
            found = self._zero_denominator(expression)
            if found is not None:
                return found
        return None

    @staticmethod
    def _undefined_denominator_steps(
        query: Query,
        denominator: Expression,
    ) -> tuple[SolutionStep, ...]:
        """Show the denominator collapsing to zero, then the undefined conclusion."""
        return (
            SolutionStep(
                rule="Simplify the denominator",
                before=denominator,
                after=Number(value=Fraction(0)),
                explanation=(
                    "Collect and cancel terms inside the denominator. Every remaining term "
                    "cancels, so the denominator reduces to exactly zero."
                ),
                verification=Verification(
                    method=VerificationMethod.EXACT_ARITHMETIC,
                    detail="The denominator simplifies to exactly zero.",
                ),
                notes=(
                    StepNote(
                        label="Zero denominator",
                        expression=Relation(
                            operator=RelationOperator.EQUAL,
                            left=denominator,
                            right=Number(value=Fraction(0)),
                        ),
                    ),
                ),
            ),
            SolutionStep(
                rule="Conclude the expression is undefined",
                before=query_expression(query),
                after=OpaqueExpression(text="undefined"),
                explanation=(
                    "Division by zero has no value, so the whole expression is undefined. "
                    "Without a defined expression there is no solution set to report."
                ),
                verification=Verification(
                    method=VerificationMethod.BACKEND_IDENTITY,
                    detail="No value can be assigned to an expression with a zero denominator.",
                ),
            ),
        )

    def _simplifies_to_zero(self, expression: Expression) -> bool:
        """Check whether an expression is identically zero in the backend."""
        try:
            value = self._converter.to_sympy(expression)
            if value == sp.Integer(0):
                return True
            return expressions_equivalent(value, sp.Integer(0))
        except (BackendError, QueryError, TypeError, ValueError):
            return False

    def _simplifies_to_negative(self, expression: Expression) -> bool:
        """Check whether a numeric exponent is definitely negative."""
        try:
            return sp.simplify(self._converter.to_sympy(expression)).is_negative is True
        except (BackendError, QueryError, TypeError, ValueError):
            return False

    @staticmethod
    def _divergence_reasons(operation: Operation) -> dict[DivergenceKind, str]:
        if operation is Operation.LIMIT:
            return {
                DivergenceKind.POSITIVE_INFINITY: "The limit is +infinity.",
                DivergenceKind.NEGATIVE_INFINITY: "The limit is -infinity.",
                DivergenceKind.NONFINITE: (
                    "The one-sided limits disagree or do not settle on a finite value, so the "
                    "limit does not exist."
                ),
            }
        noun = "series" if operation is Operation.SUM else "improper integral"
        return {
            DivergenceKind.POSITIVE_INFINITY: f"The {noun} diverges to +infinity.",
            DivergenceKind.NEGATIVE_INFINITY: f"The {noun} diverges to -infinity.",
            DivergenceKind.NONFINITE: f"The {noun} does not converge to a finite value.",
        }

    def _undefined_sum_result(self, query: Query) -> UndefinedResult | None:
        if query.operation is not Operation.SUM or len(query.arguments) != _BOUNDED_OPERATION_ARITY:
            return None
        expression_node, variable_node, lower_node, upper_node = query.arguments
        if not isinstance(variable_node, Symbol):
            return None
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        if not isinstance(variable, sp.Symbol):
            return None
        lower = self._converter.to_sympy(lower_node)
        upper = self._converter.to_sympy(upper_node)
        undefined = find_undefined_summation(expression, variable, lower, upper)
        if undefined is None:
            return None
        steps = tuple(self._renderer.solution_step(item) for item in undefined.steps)
        return UndefinedResult(
            query=query,
            reason=(
                f"The summand is undefined at {variable} = {undefined.index} because its "
                "denominator is zero."
            ),
            steps=steps,
        )

    def _nonexistent_limit_result(self, query: Query) -> DivergentResult | None:
        """Turn a backend refusal to evaluate a two-sided limit into a structured result."""
        if query.operation is not Operation.LIMIT or len(query.arguments) != 3:
            return None
        if not isinstance(query.arguments[1], Symbol):
            return None
        steps = self._steps.detailed_limit_steps(query, sp.nan)
        if not steps:
            return None
        return DivergentResult(
            query=query,
            kind=DivergenceKind.NONFINITE,
            reason=self._divergence_reasons(query.operation)[DivergenceKind.NONFINITE],
            steps=steps,
        )

    def _divergence_kind(
        self,
        query: Query,
        backend_value: object,
        detailed_steps: tuple[SolutionStep, ...],
    ) -> DivergenceKind | None:
        """Classify an operation whose non-convergence was established."""
        if query.operation is Operation.SUM and len(query.arguments) == _BOUNDED_OPERATION_ARITY:
            return self._sum_divergence_kind(query, backend_value, detailed_steps)
        if (
            query.operation is Operation.INTEGRATE
            and len(query.arguments) == _DEFINITE_INTEGRAL_ARITY
        ):
            return self._integral_divergence_kind(backend_value, detailed_steps)
        if query.operation is Operation.LIMIT and len(query.arguments) in {3, 4}:
            return self._limit_divergence_kind(backend_value)
        return None

    @staticmethod
    def _limit_divergence_kind(backend_value: object) -> DivergenceKind | None:
        """Classify a limit whose two-sided value is non-finite or unsettled."""
        if not isinstance(backend_value, sp.Basic):
            return None
        if backend_value in {sp.nan, sp.zoo}:
            return DivergenceKind.NONFINITE
        if backend_value.has(getattr(sp, "AccumBounds")):
            return DivergenceKind.NONFINITE
        return None

    def _sum_divergence_kind(
        self,
        query: Query,
        backend_value: object,
        detailed_steps: tuple[SolutionStep, ...],
    ) -> DivergenceKind | None:
        """Classify a divergent series from its backend value or verified proof."""
        if backend_value == sp.oo:
            return DivergenceKind.POSITIVE_INFINITY
        if backend_value == -sp.oo:
            return DivergenceKind.NEGATIVE_INFINITY
        if backend_value in {sp.nan, sp.zoo}:
            return DivergenceKind.NONFINITE
        if contains_unevaluated_operation(backend_value) and self._is_divergent_geometric_sum(
            query
        ):
            return DivergenceKind.NONFINITE
        return self._displayed_divergence_kind(detailed_steps)

    def _integral_divergence_kind(
        self,
        backend_value: object,
        detailed_steps: tuple[SolutionStep, ...],
    ) -> DivergenceKind | None:
        """Classify a divergent definite integral from its value or endpoint proof."""
        if backend_value == sp.oo:
            return DivergenceKind.POSITIVE_INFINITY
        if backend_value == -sp.oo:
            return DivergenceKind.NEGATIVE_INFINITY
        if backend_value in {sp.nan, sp.zoo}:
            return DivergenceKind.NONFINITE
        return self._displayed_divergence_kind(detailed_steps)

    @staticmethod
    def _displayed_divergence_kind(
        detailed_steps: tuple[SolutionStep, ...],
    ) -> DivergenceKind | None:
        """Read a directional infinity established by the final verified step."""
        final_expression = detailed_steps[-1].after if detailed_steps else None
        if final_expression == Constant(name=ConstantName.INFINITY):
            return DivergenceKind.POSITIVE_INFINITY
        negative_infinity = UnaryExpression(
            operator=UnaryOperator.NEGATIVE,
            operand=Constant(name=ConstantName.INFINITY),
        )
        if final_expression == negative_infinity:
            return DivergenceKind.NEGATIVE_INFINITY
        return None

    def _is_divergent_geometric_sum(self, query: Query) -> bool:
        expression_node, variable_node, lower_node, upper_node = query.arguments
        if not isinstance(variable_node, Symbol):
            return False
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        lower = self._converter.to_sympy(lower_node)
        upper = self._converter.to_sympy(upper_node)
        if (
            not isinstance(variable, sp.Symbol)
            or lower.is_integer is not True
            or upper != sp.oo
            or not expression.is_Pow
            or len(expression.args) != _EQUATION_ARITY
        ):
            return False
        ratio, exponent = expression.args
        if exponent != variable:
            return False
        return sp.simplify(sp.Ge(sp.Abs(ratio), sp.Integer(1))) == sp.true

    @staticmethod
    def _non_exact_reason(backend_value: object) -> str | None:
        """Reject backend placeholders as exact answers."""
        if contains_unevaluated_operation(backend_value):
            return (
                "The symbolic backend could not evaluate this operation exactly. "
                "The unevaluated operation has been kept out of the answer."
            )
        return None

    def _identity_equation_result(self, query: Query) -> ExactResult | None:
        """Return an exact universal solution when the original equation is an identity."""
        if query.operation is not Operation.SOLVE or len(query.arguments) != _EQUATION_ARITY:
            return None
        equation_expression, variable_expression = query.arguments
        if (
            not isinstance(equation_expression, Relation)
            or equation_expression.operator is not RelationOperator.EQUAL
            or not isinstance(variable_expression, Symbol)
            or not self._identity_domain_supported(equation_expression)
        ):
            return None
        equation = self._converter.to_sympy(equation_expression)
        variable = self._converter.to_sympy(variable_expression)
        if not isinstance(equation, sp.Equality) or not isinstance(variable, sp.Symbol):
            return None
        if not equation.free_symbols <= {variable}:
            return None
        if not expressions_equivalent(equation.lhs, equation.rhs):
            return None
        denominator_expressions = domain_denominators(equation_expression)
        denominators = tuple(
            self._converter.to_sympy(denominator) for denominator in denominator_expressions
        )
        restrictions = original_denominator_restrictions(denominators, variable)
        exclusions = tuple(
            self._converter.from_sympy(value) for value in restrictions.excluded_values
        )
        constraints = tuple(
            StepConstraint(
                explanation=item.explanation,
                expression=self._renderer.derivation_expression(item.expression),
            )
            for item in restrictions.displayed
        )
        zero = self._converter.from_sympy(sp.Integer(0))
        zero_equation = Relation(
            operator=RelationOperator.EQUAL,
            left=zero,
            right=zero,
        )
        conclusion = FunctionCall(
            name=Identifier("all_real"),
            arguments=(variable_expression,),
        )
        domain_qualifier = (
            " The original denominators must stay nonzero, so the stated exclusions remain "
            "outside the domain."
            if constraints
            else ""
        )
        return ExactResult(
            query=query,
            value=UniversalSolutionValue(
                domain=SolutionDomain.REAL,
                exclusions=exclusions,
            ),
            steps=(
                SolutionStep(
                    rule="Subtract matching sides",
                    before=equation_expression,
                    after=zero_equation,
                    explanation=(
                        "Subtract the right side from both sides. The two sides are identical "
                        f"wherever the equation is defined, leaving 0 = 0.{domain_qualifier}"
                    ),
                    verification=Verification(
                        method=VerificationMethod.BACKEND_IDENTITY,
                        detail=(
                            "Exact symbolic simplification of the difference between both sides "
                            "gives zero."
                        ),
                    ),
                    introduced_constraints=constraints,
                ),
                SolutionStep(
                    rule="Conclude every admissible real value is a solution",
                    before=zero_equation,
                    after=conclusion,
                    explanation=(
                        "The identity 0 = 0 holds for every admissible real value, so every "
                        "admissible real value is a solution."
                    ),
                    verification=Verification(
                        method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                        detail=(
                            "The reduced identity holds for every value allowed by the domain "
                            "restrictions."
                        ),
                    ),
                ),
            ),
        )

    @staticmethod
    def _identity_domain_supported(expression: object) -> bool:
        """Limit universal answers to expressions whose real domain is denominator-only."""
        if isinstance(expression, Number | Symbol):
            return True
        if isinstance(expression, ApproximateNumber | OpaqueExpression):
            return False
        if isinstance(expression, Constant):
            return expression.name is not ConstantName.INFINITY
        if isinstance(expression, UnaryExpression):
            return (
                expression.operator is not UnaryOperator.FACTORIAL
                and SympyBackend._identity_domain_supported(expression.operand)
            )
        if isinstance(expression, BinaryExpression):
            if expression.operator is BinaryOperator.POWER and not (
                isinstance(expression.right, Number) and expression.right.value.denominator == 1
            ):
                return False
            return SympyBackend._identity_domain_supported(
                expression.left
            ) and SympyBackend._identity_domain_supported(expression.right)
        if isinstance(expression, FunctionCall):
            return False
        if isinstance(expression, Relation):
            return SympyBackend._identity_domain_supported(
                expression.left
            ) and SympyBackend._identity_domain_supported(expression.right)
        if isinstance(expression, SequenceExpression):
            return all(SympyBackend._identity_domain_supported(item) for item in expression.items)
        return False

    def _identity_equation_reason(self, query: Query) -> str | None:
        if query.operation is not Operation.SOLVE or len(query.arguments) != _EQUATION_ARITY:
            return None
        equation, variable = query.arguments
        if not isinstance(equation, Relation) or not isinstance(variable, Symbol):
            return None
        difference_is_zero = expressions_equivalent(
            self._converter.to_sympy(equation.left),
            self._converter.to_sympy(equation.right),
        )
        if not difference_is_zero:
            return None
        if equation.operator is not RelationOperator.EQUAL:
            return (
                "This relation compares identical expressions. Its truth set may be universal "
                "or empty, and StepSolver cannot yet represent it safely."
            )
        return (
            "This equation is an identity, but StepSolver cannot yet represent its full "
            "real-domain conditions safely. It has not been reported as all real numbers."
        )

    def _solve_contour(self, query: Query) -> SolveResult:
        expect_arity(query, 6)
        integrand, variable_expression, path, parameter_expression, lower, upper = query.arguments
        variable = expect_symbol(variable_expression, role="contour variable")
        parameter = expect_symbol(parameter_expression, role="path parameter")
        substituted = substitute(integrand, variable, path)
        path_derivative = FunctionCall(
            name=Identifier(Operation.DIFFERENTIATE.value), arguments=(path, parameter)
        )
        transformed_integrand = BinaryExpression(
            operator=BinaryOperator.MULTIPLY,
            left=substituted,
            right=path_derivative,
        )
        transformed_query = Query(
            operation=Operation.INTEGRATE,
            arguments=(transformed_integrand, parameter, lower, upper),
            source=query.source,
        )
        backend_integrand = self._converter.to_sympy(integrand)
        backend_variable = self._converter.to_sympy(variable)
        backend_path = self._converter.to_sympy(path)
        backend_parameter = self._converter.to_sympy(parameter)
        backend_lower = self._converter.to_sympy(lower)
        backend_upper = self._converter.to_sympy(upper)
        if not isinstance(backend_variable, sp.Symbol) or not isinstance(
            backend_parameter, sp.Symbol
        ):
            message = "contour variables did not convert to symbols"
            raise BackendError(message)
        circle = evaluate_rational_circle(
            ContourExpression(
                integrand=backend_integrand,
                variable=backend_variable,
                path=backend_path,
                parameter=backend_parameter,
                lower=backend_lower,
                upper=backend_upper,
            )
        )
        if circle is not None:
            residue_notes = tuple(
                StepNote(
                    label=f"Residue at {pole}",
                    expression=self._converter.from_sympy(residue),
                )
                for pole, residue in circle.enclosed_residues
            )
            residue_sum_expression = self._converter.from_sympy(circle.residue_sum)
            residue_factor_expression = self._converter.from_sympy(
                2 * sp.pi * sp.I * circle.winding_number
            )
            residue_theorem_expression = BinaryExpression(
                operator=BinaryOperator.MULTIPLY,
                left=residue_factor_expression,
                right=residue_sum_expression,
            )
            transformation = FunctionCall(
                name=Identifier(Operation.INTEGRATE.value),
                arguments=(transformed_integrand, parameter, lower, upper),
            )
            contour_steps = (
                SolutionStep(
                    rule="parameterize contour",
                    before=query_expression(query),
                    after=transformation,
                    explanation=(
                        "Substitute the circular path and multiply by its parameter derivative."
                    ),
                    verification=Verification(
                        method=VerificationMethod.SUBSTITUTION,
                        detail=("The contour identity dz = gamma'(t) dt was applied structurally."),
                    ),
                ),
                SolutionStep(
                    rule="sum enclosed residues",
                    before=transformation,
                    after=residue_sum_expression,
                    explanation=("Find the poles inside the circle and add their residues."),
                    notes=residue_notes,
                    verification=Verification(
                        method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                        detail="Each listed residue was computed from the rational integrand.",
                    ),
                ),
                SolutionStep(
                    rule="apply the residue theorem",
                    before=residue_theorem_expression,
                    after=self._converter.from_sympy(circle.value),
                    explanation=(
                        "Multiply the enclosed-residue sum by 2*pi*i and the contour's "
                        "winding number."
                    ),
                    verification=Verification(
                        method=VerificationMethod.EXACT_ARITHMETIC,
                        detail="The residue theorem factor and residue sum were combined exactly.",
                    ),
                ),
            )
            return ExactResult(
                query=query,
                value=self._converter.to_value(circle.value),
                steps=contour_steps,
            )
        backend_result = self._executor.execute(transformed_query)
        non_exact_reason = self._non_exact_reason(backend_result)
        if non_exact_reason is not None:
            return UnsolvedResult(query=query, reason=non_exact_reason, steps=())
        value = self._converter.to_value(backend_result)
        transformation = FunctionCall(
            name=Identifier(Operation.INTEGRATE.value),
            arguments=(transformed_integrand, parameter, lower, upper),
        )
        fallback_steps = (
            SolutionStep(
                rule="parameterize contour",
                before=query_expression(query),
                after=transformation,
                explanation="Substitute the path and multiply by its parameter derivative.",
                verification=Verification(
                    method=VerificationMethod.SUBSTITUTION,
                    detail="The contour identity dz = gamma'(t) dt was applied structurally.",
                ),
            ),
            SolutionStep(
                rule="evaluate parameter integral",
                before=transformation,
                after=self._converter.step_expression(backend_result),
                explanation="Evaluate the resulting definite integral over the path parameter.",
                verification=Verification(
                    method=VerificationMethod.BACKEND_IDENTITY,
                    detail="SymPy evaluated the exact parameterized integral.",
                ),
            ),
        )
        return ExactResult(query=query, value=value, steps=fallback_steps)
