"""SymPy implementation of the StepSolver symbolic backend."""

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TypeGuard, assert_never

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
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.errors import BackendError, ParseError, QueryError
from stepsolver.parser import parse_expression
from stepsolver.results import (
    BooleanValue,
    DivergenceKind,
    DivergentResult,
    ExactResult,
    MappingEntry,
    MappingValue,
    MathValue,
    MatrixValue,
    ScalarValue,
    SequenceValue,
    SolutionStep,
    SolveResult,
    StepNote,
    UnsolvedResult,
    Verification,
    VerificationMethod,
)
from stepsolver.sympy_derivation import (
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendDifferential,
    BackendEvaluationAtBounds,
    BackendExpression,
    BackendIdentity,
    BackendIntegral,
    BackendIntegrationByPartsRule,
    BackendLimit,
    BackendNotEqual,
    BackendProduct,
    BackendQuadraticSolutions,
    BackendQuotient,
    BackendSum,
    derive_basic_antiderivative,
    derive_constant_multiple_integral,
    derive_definite_integral,
    derive_derivative,
    derive_dirichlet_integral,
    derive_function_substitution_integral,
    derive_gaussian_antiderivative,
    derive_improper_integral,
    derive_integration_by_parts,
    derive_inverse_hyperbolic_integral,
    derive_limit,
    derive_log_derivative_integral,
    derive_partial_fraction_integral,
    derive_polynomial_equation,
    derive_polynomial_sum_integral,
    derive_reciprocal_quadratic_integral,
    derive_shifted_semicircle_integral,
    derive_square_root_rational_integral,
    derive_trigonometric_power_integral,
)

_INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")
_RATIONAL_PATTERN = re.compile(r"^-?[0-9]+/[0-9]+$")
_DECIMAL_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]+$")
_UNEVALUATED_OPERATION_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        ("sympy.concrete.products", "Product"),
        ("sympy.concrete.summations", "Sum"),
        ("sympy.core.function", "Derivative"),
        ("sympy.integrals.integrals", "Integral"),
        ("sympy.series.limits", "Limit"),
    }
)


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _contains_unevaluated_operation(value: object) -> bool:
    """Return whether a backend result still contains an unevaluated operation."""
    if isinstance(value, sp.Basic):
        operation_type = (type(value).__module__, type(value).__name__)
        return operation_type in _UNEVALUATED_OPERATION_TYPES or any(
            _contains_unevaluated_operation(argument) for argument in value.args
        )
    if _is_object_mapping(value):
        return any(
            _contains_unevaluated_operation(key) or _contains_unevaluated_operation(item)
            for key, item in value.items()
        )
    if _is_object_sequence(value):
        return any(_contains_unevaluated_operation(item) for item in value)
    return False


def _query_expression(query: Query) -> FunctionCall:
    return FunctionCall(name=Identifier(query.operation.value), arguments=query.arguments)


def _derive_indefinite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    strategies = (
        derive_basic_antiderivative,
        derive_log_derivative_integral,
        derive_polynomial_sum_integral,
        derive_constant_multiple_integral,
        derive_function_substitution_integral,
        derive_square_root_rational_integral,
        derive_shifted_semicircle_integral,
        derive_inverse_hyperbolic_integral,
        derive_trigonometric_power_integral,
        derive_integration_by_parts,
        derive_gaussian_antiderivative,
        derive_reciprocal_quadratic_integral,
        derive_partial_fraction_integral,
    )
    for strategy in strategies:
        derivation = strategy(integrand, variable, result)
        if derivation:
            return derivation
    return ()


def _derive_definite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    derivation = derive_dirichlet_integral(
        integrand,
        variable,
        lower,
        upper,
        result,
    )
    if derivation:
        return derivation
    derivation = derive_improper_integral(integrand, variable, lower, upper, result)
    if derivation:
        return derivation
    return derive_definite_integral(integrand, variable, lower, upper, result)


def _evaluate_indefinite_integral(
    integrand: sp.Basic,
    variable: sp.Basic,
) -> sp.Basic:
    integration_constant = sp.Symbol("C")
    antiderivative = sp.integrate(integrand, variable)
    if isinstance(variable, sp.Symbol):
        sine_square = sp.sin(variable) ** 2
        cosine_square = sp.cos(variable) ** 2
        if sp.simplify(integrand - sine_square) == sp.Integer(0):
            return sp.Add(
                variable / 2,
                -sp.sin(2 * variable) / 4,
                integration_constant,
                evaluate=False,
            )
        if sp.simplify(integrand - cosine_square) == sp.Integer(0):
            return sp.Add(
                variable / 2,
                sp.sin(2 * variable) / 4,
                integration_constant,
                evaluate=False,
            )
        if integrand.is_rational_function(variable) and not antiderivative.has(sp.log):
            decomposition = sp.apart(integrand, variable)
            terms = tuple(decomposition.as_ordered_terms())
            if str(decomposition) != str(integrand) and len(terms) > 1:
                return sp.Add(
                    *(sp.integrate(term, variable) for term in terms),
                    integration_constant,
                    evaluate=False,
                )
    return antiderivative + integration_constant


def _expect_arity(query: Query, *allowed: int) -> None:
    if len(query.arguments) not in allowed:
        choices = ", ".join(str(value) for value in allowed)
        raise QueryError(
            f"{query.operation.value} expects {choices} argument(s), got {len(query.arguments)}"
        )


def _expect_symbol(expression: Expression, *, role: str) -> Symbol:
    if not isinstance(expression, Symbol):
        raise QueryError(f"{role} must be a symbol")
    return expression


def _expect_integer(expression: Expression, *, role: str) -> int:
    if not isinstance(expression, Number) or expression.value.denominator != 1:
        raise QueryError(f"{role} must be an integer")
    return expression.value.numerator


def _substitute(expression: Expression, symbol: Symbol, replacement: Expression) -> Expression:
    if isinstance(expression, Symbol):
        return replacement if expression == symbol else expression
    if isinstance(expression, Number | ApproximateNumber | Constant | OpaqueExpression):
        return expression
    if isinstance(expression, UnaryExpression):
        return UnaryExpression(
            operator=expression.operator,
            operand=_substitute(expression.operand, symbol, replacement),
        )
    if isinstance(expression, BinaryExpression):
        return BinaryExpression(
            operator=expression.operator,
            left=_substitute(expression.left, symbol, replacement),
            right=_substitute(expression.right, symbol, replacement),
        )
    if isinstance(expression, FunctionCall):
        return FunctionCall(
            name=expression.name,
            arguments=tuple(
                _substitute(argument, symbol, replacement) for argument in expression.arguments
            ),
        )
    if isinstance(expression, Relation):
        return Relation(
            operator=expression.operator,
            left=_substitute(expression.left, symbol, replacement),
            right=_substitute(expression.right, symbol, replacement),
        )
    return SequenceExpression(
        items=tuple(_substitute(item, symbol, replacement) for item in expression.items)
    )


class SympyBackend:
    """Solve StepSolver queries using an isolated SymPy adapter."""

    def solve(self, query: Query) -> SolveResult:
        """Solve a validated query and translate the result to StepSolver values."""
        try:
            return self._solve(query)
        except (BackendError, NotImplementedError, QueryError, TypeError, ValueError) as error:
            return UnsolvedResult(query=query, reason=str(error), steps=())

    def _solve(self, query: Query) -> SolveResult:
        if query.operation is Operation.CONTOUR_INTEGRATE:
            return self._solve_contour(query)
        identity_reason = self._identity_equation_reason(query)
        if identity_reason is not None:
            return UnsolvedResult(query=query, reason=identity_reason, steps=())
        integral_domain_reason = self._integral_domain_reason(query)
        if integral_domain_reason is not None:
            return UnsolvedResult(query=query, reason=integral_domain_reason, steps=())
        backend_value = self._execute(query)
        detailed_steps = self._detailed_steps(query, backend_value)
        divergence = self._divergence_kind(query, backend_value, detailed_steps)
        if divergence is not None:
            reasons = {
                DivergenceKind.POSITIVE_INFINITY: ("The improper integral diverges to +infinity."),
                DivergenceKind.NEGATIVE_INFINITY: ("The improper integral diverges to -infinity."),
                DivergenceKind.NONFINITE: (
                    "The improper integral does not converge to a finite value."
                ),
            }
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
        value = self._to_value(backend_value)
        if detailed_steps:
            return ExactResult(query=query, value=value, steps=detailed_steps)
        after = self._step_expression(backend_value)
        step = SolutionStep(
            rule="Compute exact result",
            before=_query_expression(query),
            after=after,
            explanation=(
                f"Evaluate the {query.operation.value} operation with the symbolic backend."
            ),
            verification=self._verify_result(query, backend_value),
        )
        return ExactResult(query=query, value=value, steps=(step,))

    def _divergence_kind(
        self,
        query: Query,
        backend_value: object,
        detailed_steps: tuple[SolutionStep, ...],
    ) -> DivergenceKind | None:
        """Classify a definite integral whose non-convergence was established."""
        if query.operation is Operation.INTEGRATE and len(query.arguments) == 4:
            final_expression = detailed_steps[-1].after if detailed_steps else None
            if backend_value == sp.oo or final_expression == Constant(name=ConstantName.INFINITY):
                return DivergenceKind.POSITIVE_INFINITY
            negative_infinity = UnaryExpression(
                operator=UnaryOperator.NEGATIVE,
                operand=Constant(name=ConstantName.INFINITY),
            )
            if backend_value == -sp.oo or final_expression == negative_infinity:
                return DivergenceKind.NEGATIVE_INFINITY
            if backend_value in {sp.nan, sp.zoo}:
                return DivergenceKind.NONFINITE
        return None

    def _non_exact_reason(self, backend_value: object) -> str | None:
        """Reject backend placeholders as exact answers."""
        if _contains_unevaluated_operation(backend_value):
            return (
                "The symbolic backend could not evaluate this operation exactly. "
                "The unevaluated operation has been kept out of the answer."
            )
        return None

    def _integral_domain_reason(self, query: Query) -> str | None:
        if query.operation is not Operation.INTEGRATE or len(query.arguments) != 2:
            return None
        variable_node = query.arguments[1]
        if not isinstance(variable_node, Symbol):
            return None
        integrand = self._to_sympy(query.arguments[0])
        variable = self._to_sympy(variable_node)
        if not isinstance(variable, sp.Symbol) or not integrand.is_rational_function(variable):
            return None
        _numerator, denominator = sp.fraction(sp.together(integrand))
        try:
            denominator_polynomial = sp.Poly(denominator, variable)
        except sp.PolynomialError:
            return None
        roots = sp.solve(denominator_polynomial.as_expr(), variable)
        if not _is_object_sequence(roots):
            return None
        has_real_pole = any(isinstance(root, sp.Basic) and root.is_real is True for root in roots)
        if not has_real_pole or not sp.integrate(integrand, variable).has(sp.log):
            return None
        return (
            "This antiderivative requires logarithms of absolute values and explicit domain "
            "intervals around real poles. The current result model cannot represent those "
            "domain conditions safely yet."
        )

    def _identity_equation_reason(self, query: Query) -> str | None:
        if query.operation is not Operation.SOLVE or len(query.arguments) != 2:
            return None
        equation, variable = query.arguments
        if not isinstance(equation, Relation) or not isinstance(variable, Symbol):
            return None
        difference = sp.simplify(self._to_sympy(equation.left) - self._to_sympy(equation.right))
        if difference != sp.Integer(0):
            return None
        return (
            "This equation is true for every value in its domain. The current result model "
            "cannot yet represent a universal solution set together with possible domain "
            "exclusions."
        )

    def _detailed_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if query.operation is Operation.SOLVE:
            return self._detailed_equation_steps(query, backend_value)
        if query.operation is Operation.DIFFERENTIATE:
            return self._detailed_derivative_steps(query, backend_value)
        if query.operation is Operation.INTEGRATE:
            return self._detailed_integral_steps(query, backend_value)
        if query.operation is Operation.LIMIT:
            return self._detailed_limit_steps(query, backend_value)
        return ()

    def _detailed_limit_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if len(query.arguments) not in {3, 4} or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node, point_node = query.arguments[:3]
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._to_sympy(expression_node)
        variable = self._to_sympy(variable_node)
        point = self._to_sympy(point_node)
        if not isinstance(variable, sp.Symbol):
            return ()
        direction: str | None = None
        if len(query.arguments) == 4 and isinstance(query.arguments[3], Symbol):
            direction = {"left": "-", "right": "+"}.get(query.arguments[3].name)
        try:
            derivation = derive_limit(
                expression,
                variable,
                point,
                direction,
                backend_value,
            )
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._solution_step(item) for item in derivation)

    def _detailed_equation_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if len(query.arguments) != 2:
            return ()
        equation_expression, variable_expression = query.arguments
        if not isinstance(equation_expression, Relation) or not isinstance(
            variable_expression,
            Symbol,
        ):
            return ()
        equation = self._to_sympy(equation_expression)
        variable = self._to_sympy(variable_expression)
        if not isinstance(equation, sp.Equality) or not isinstance(variable, sp.Symbol):
            return ()
        roots = self._solution_roots(backend_value, variable)
        try:
            derivation = derive_polynomial_equation(equation, variable, roots)
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._solution_step(item) for item in derivation)

    def _detailed_derivative_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if len(query.arguments) != 2 or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node = query.arguments
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._to_sympy(expression_node)
        variable = self._to_sympy(variable_node)
        if not isinstance(variable, sp.Symbol):
            return ()
        try:
            derivation = derive_derivative(expression, variable, backend_value)
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._solution_step(item) for item in derivation)

    def _detailed_integral_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if len(query.arguments) not in {2, 4} or not isinstance(backend_value, sp.Basic):
            return ()
        integrand_expression, variable_expression = query.arguments[:2]
        if not isinstance(variable_expression, Symbol):
            return ()
        integrand = self._to_sympy(integrand_expression)
        variable = self._to_sympy(variable_expression)
        if not isinstance(variable, sp.Symbol):
            return ()
        if len(query.arguments) == 2:
            try:
                derivation = _derive_indefinite_integral(integrand, variable, backend_value)
            except (sp.PolynomialError, TypeError, ValueError):
                return ()
        else:
            lower = self._to_sympy(query.arguments[2])
            upper = self._to_sympy(query.arguments[3])
            try:
                derivation = _derive_definite_integral(
                    integrand, variable, lower, upper, backend_value
                )
            except (sp.PolynomialError, TypeError, ValueError):
                return ()
        return tuple(self._solution_step(item) for item in derivation)

    def _solution_roots(
        self,
        backend_value: object,
        variable: sp.Symbol,
    ) -> tuple[sp.Basic, ...]:
        if not _is_object_sequence(backend_value):
            return ()
        roots: list[sp.Basic] = []
        for solution in backend_value:
            if not _is_object_mapping(solution):
                return ()
            root = solution.get(variable)
            if not isinstance(root, sp.Basic):
                return ()
            roots.append(root)
        return tuple(roots)

    def _solution_step(self, step: BackendDerivationStep) -> SolutionStep:
        return SolutionStep(
            rule=step.rule,
            before=self._derivation_expression(step.before),
            after=self._derivation_expression(step.after),
            explanation=step.explanation,
            verification=Verification(
                method=step.verification_method,
                detail=step.verification_detail,
            ),
            notes=tuple(
                StepNote(
                    label=note.label,
                    expression=self._derivation_expression(note.expression),
                )
                for note in step.notes
            ),
        )

    def _derivation_expression(
        self,
        value: BackendExpression,
    ) -> Expression:
        if isinstance(value, BackendIdentity):
            return Relation(
                operator=RelationOperator.EQUAL,
                left=self._derivation_expression(value.left),
                right=self._derivation_expression(value.right),
            )
        if isinstance(value, BackendIntegrationByPartsRule):
            return FunctionCall(name=Identifier("integration_by_parts_rule"), arguments=())
        if isinstance(value, BackendQuadraticSolutions):
            return FunctionCall(
                name=Identifier("quadratic_solutions"),
                arguments=(
                    self._from_sympy(value.variable),
                    self._derivation_expression(value.negative_numerator),
                    self._derivation_expression(value.positive_numerator),
                    self._derivation_expression(value.denominator),
                ),
            )
        if isinstance(value, BackendNotEqual):
            return Relation(
                operator=RelationOperator.NOT_EQUAL,
                left=self._derivation_expression(value.left),
                right=self._derivation_expression(value.right),
            )
        if isinstance(value, BackendSum):
            expressions = tuple(self._derivation_expression(term) for term in value.terms)
            first, *remaining = expressions
            result = first
            for expression in remaining:
                result = BinaryExpression(
                    operator=BinaryOperator.ADD,
                    left=result,
                    right=expression,
                )
            return result
        if isinstance(value, BackendProduct):
            expressions = tuple(self._derivation_expression(factor) for factor in value.factors)
            first, *remaining = expressions
            result = first
            for expression in remaining:
                result = BinaryExpression(
                    operator=BinaryOperator.MULTIPLY,
                    left=result,
                    right=expression,
                )
            return result
        if isinstance(value, BackendQuotient):
            return BinaryExpression(
                operator=BinaryOperator.DIVIDE,
                left=self._derivation_expression(value.numerator),
                right=self._derivation_expression(value.denominator),
            )
        if isinstance(value, BackendDifference):
            return BinaryExpression(
                operator=BinaryOperator.SUBTRACT,
                left=self._derivation_expression(value.left),
                right=self._derivation_expression(value.right),
            )
        if isinstance(value, BackendDifferential):
            differential = FunctionCall(
                name=Identifier("differential"),
                arguments=(self._from_sympy(value.variable),),
            )
            if value.coefficient is None:
                return differential
            return BinaryExpression(
                operator=BinaryOperator.MULTIPLY,
                left=self._from_sympy(value.coefficient),
                right=differential,
            )
        if isinstance(value, BackendDerivative):
            return FunctionCall(
                name=Identifier(Operation.DIFFERENTIATE.value),
                arguments=(
                    self._from_sympy(value.expression),
                    self._from_sympy(value.variable),
                ),
            )
        if isinstance(value, BackendEvaluationAtBounds):
            return FunctionCall(
                name=Identifier("evaluate_at_bounds"),
                arguments=(
                    self._from_sympy(value.expression),
                    self._from_sympy(value.variable),
                    self._from_sympy(value.lower),
                    self._from_sympy(value.upper),
                ),
            )
        if isinstance(value, BackendLimit):
            limit_arguments: tuple[Expression, ...] = (
                self._derivation_expression(value.expression),
                self._from_sympy(value.variable),
                self._from_sympy(value.point),
            )
            if value.direction is not None:
                direction = "right" if value.direction == "+" else "left"
                limit_arguments = (*limit_arguments, Symbol(name=Identifier(direction)))
            return FunctionCall(
                name=Identifier(Operation.LIMIT.value),
                arguments=limit_arguments,
            )
        if isinstance(value, BackendIntegral):
            integral_arguments: tuple[Expression, ...] = (
                self._from_sympy(value.integrand),
                self._from_sympy(value.variable),
            )
            if value.lower is not None and value.upper is not None:
                integral_arguments = (
                    *integral_arguments,
                    self._from_sympy(value.lower),
                    self._from_sympy(value.upper),
                )
            integral = FunctionCall(
                name=Identifier(Operation.INTEGRATE.value),
                arguments=integral_arguments,
            )
            if value.coefficient is None:
                return integral
            return BinaryExpression(
                operator=BinaryOperator.MULTIPLY,
                left=self._from_sympy(value.coefficient),
                right=integral,
            )
        if isinstance(value, tuple):
            return SequenceExpression(items=tuple(self._from_sympy(item) for item in value))
        return self._from_sympy(value)

    def _execute(self, query: Query) -> object:
        operation = query.operation
        arguments = query.arguments
        match operation:
            case Operation.SIMPLIFY:
                _expect_arity(query, 1)
                return sp.simplify(self._to_sympy(arguments[0]))
            case Operation.EXPAND:
                _expect_arity(query, 1)
                return sp.expand(self._to_sympy(arguments[0]))
            case Operation.FACTOR:
                _expect_arity(query, 1)
                return sp.factor(self._to_sympy(arguments[0]))
            case Operation.CANCEL:
                _expect_arity(query, 1)
                return sp.cancel(self._to_sympy(arguments[0]))
            case Operation.APART:
                _expect_arity(query, 1, 2)
                variable = self._to_sympy(arguments[1]) if len(arguments) == 2 else None
                return sp.apart(self._to_sympy(arguments[0]), variable)
            case Operation.SOLVE:
                _expect_arity(query, 2)
                equations = self._equations(arguments[0])
                symbols = self._symbols(arguments[1])
                return sp.solve(equations, *symbols, dict=True)
            case Operation.SOLVE_INEQUALITY:
                _expect_arity(query, 2)
                symbol = _expect_symbol(arguments[1], role="inequality variable")
                backend_symbol = self._to_sympy(symbol)
                if not isinstance(backend_symbol, sp.Symbol):
                    raise BackendError("symbol conversion did not produce a SymPy Symbol")
                return sp.solve_univariate_inequality(self._to_sympy(arguments[0]), backend_symbol)
            case Operation.DIFFERENTIATE:
                _expect_arity(query, 2, 3)
                order = (
                    _expect_integer(arguments[2], role="derivative order")
                    if len(arguments) == 3
                    else 1
                )
                return sp.diff(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    order,
                )
            case Operation.INTEGRATE:
                _expect_arity(query, 2, 4)
                integrand = self._to_sympy(arguments[0])
                variable = self._to_sympy(arguments[1])
                if len(arguments) == 2:
                    return _evaluate_indefinite_integral(integrand, variable)
                return sp.integrate(
                    integrand,
                    (variable, self._to_sympy(arguments[2]), self._to_sympy(arguments[3])),
                )
            case Operation.CONTOUR_INTEGRATE:
                raise BackendError("contour integrals require the dedicated execution path")
            case Operation.LIMIT:
                _expect_arity(query, 3, 4)
                direction = "+-"
                if len(arguments) == 4:
                    direction_expression = arguments[3]
                    if not isinstance(direction_expression, Symbol):
                        raise QueryError("limit direction must be left, right, or two_sided")
                    direction = {"left": "-", "right": "+", "two_sided": "+-"}.get(
                        direction_expression.name,
                        "",
                    )
                    if not direction:
                        raise QueryError("limit direction must be left, right, or two_sided")
                return sp.limit(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                    direction,
                )
            case Operation.SERIES:
                _expect_arity(query, 4)
                return sp.series(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                    _expect_integer(arguments[3], role="series order"),
                )
            case Operation.SUM:
                _expect_arity(query, 4)
                return sp.summation(
                    self._to_sympy(arguments[0]),
                    (
                        self._to_sympy(arguments[1]),
                        self._to_sympy(arguments[2]),
                        self._to_sympy(arguments[3]),
                    ),
                )
            case Operation.PRODUCT:
                _expect_arity(query, 4)
                return sp.product(
                    self._to_sympy(arguments[0]),
                    (
                        self._to_sympy(arguments[1]),
                        self._to_sympy(arguments[2]),
                        self._to_sympy(arguments[3]),
                    ),
                )
            case Operation.MATRIX:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0])
            case Operation.DETERMINANT:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0]).det()
            case Operation.INVERSE:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0]).inv()
            case Operation.RANK:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0]).rank()
            case Operation.RREF:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0]).rref()
            case Operation.EIGENVALUES:
                _expect_arity(query, 1)
                return self._matrix_from_expression(arguments[0]).eigenvals()
            case Operation.DSOLVE:
                _expect_arity(query, 2)
                return sp.dsolve(self._to_sympy(arguments[0]), self._to_sympy(arguments[1]))
            case Operation.RSOLVE:
                _expect_arity(query, 2)
                return sp.rsolve(self._to_sympy(arguments[0]), self._to_sympy(arguments[1]))
            case Operation.LAPLACE:
                _expect_arity(query, 3)
                return sp.laplace_transform(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                    noconds=True,
                )
            case Operation.INVERSE_LAPLACE:
                _expect_arity(query, 3)
                return sp.inverse_laplace_transform(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                )
            case Operation.FOURIER:
                _expect_arity(query, 3)
                return sp.fourier_transform(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                )
            case Operation.INVERSE_FOURIER:
                _expect_arity(query, 3)
                return sp.inverse_fourier_transform(
                    self._to_sympy(arguments[0]),
                    self._to_sympy(arguments[1]),
                    self._to_sympy(arguments[2]),
                )
            case Operation.GCD:
                _expect_arity(query, 2)
                return sp.gcd(self._to_sympy(arguments[0]), self._to_sympy(arguments[1]))
            case Operation.LCM:
                _expect_arity(query, 2)
                return sp.lcm(self._to_sympy(arguments[0]), self._to_sympy(arguments[1]))
            case Operation.IS_PRIME:
                _expect_arity(query, 1)
                return sp.isprime(_expect_integer(arguments[0], role="primality input"))
            case Operation.PRIME_FACTORS:
                _expect_arity(query, 1)
                return sp.factorint(_expect_integer(arguments[0], role="factorization input"))
            case Operation.BINOMIAL | Operation.COMBINATIONS:
                _expect_arity(query, 2)
                return sp.binomial(self._to_sympy(arguments[0]), self._to_sympy(arguments[1]))
            case Operation.PERMUTATIONS:
                _expect_arity(query, 2)
                n = self._to_sympy(arguments[0])
                r = self._to_sympy(arguments[1])
                return sp.factorial(n) / sp.factorial(n - r)
            case Operation.NUMERIC:
                _expect_arity(query, 1, 2)
                digits = (
                    _expect_integer(arguments[1], role="precision") if len(arguments) == 2 else 15
                )
                return sp.N(self._to_sympy(arguments[0]), digits)

    def _solve_contour(self, query: Query) -> SolveResult:
        _expect_arity(query, 6)
        integrand, variable_expression, path, parameter_expression, lower, upper = query.arguments
        variable = _expect_symbol(variable_expression, role="contour variable")
        parameter = _expect_symbol(parameter_expression, role="path parameter")
        substituted = _substitute(integrand, variable, path)
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
        backend_result = self._execute(transformed_query)
        non_exact_reason = self._non_exact_reason(backend_result)
        if non_exact_reason is not None:
            return UnsolvedResult(query=query, reason=non_exact_reason, steps=())
        value = self._to_value(backend_result)
        transformation = FunctionCall(
            name=Identifier(Operation.INTEGRATE.value),
            arguments=(transformed_integrand, parameter, lower, upper),
        )
        steps = (
            SolutionStep(
                rule="parameterize contour",
                before=_query_expression(query),
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
                after=self._step_expression(backend_result),
                explanation="Evaluate the resulting definite integral over the path parameter.",
                verification=Verification(
                    method=VerificationMethod.BACKEND_IDENTITY,
                    detail="SymPy evaluated the exact parameterized integral.",
                ),
            ),
        )
        return ExactResult(query=query, value=value, steps=steps)

    def _to_sympy(self, expression: Expression) -> sp.Basic:
        if isinstance(expression, Number):
            return sp.Rational(expression.value.numerator, expression.value.denominator)
        if isinstance(expression, ApproximateNumber):
            return sp.Float(expression.text)
        if isinstance(expression, Symbol):
            return sp.Symbol(expression.name, real=True)
        if isinstance(expression, Constant):
            constants: dict[ConstantName, sp.Expr] = {
                ConstantName.PI: sp.pi,
                ConstantName.E: sp.E,
                ConstantName.IMAGINARY: sp.I,
                ConstantName.INFINITY: sp.oo,
            }
            return constants[expression.name]
        if isinstance(expression, UnaryExpression):
            operand = self._to_sympy(expression.operand)
            if expression.operator is UnaryOperator.POSITIVE:
                return operand
            if expression.operator is UnaryOperator.NEGATIVE:
                return -operand
            if expression.operator is UnaryOperator.FACTORIAL:
                return sp.factorial(operand)
            assert_never(expression.operator)
        if isinstance(expression, BinaryExpression):
            left = self._to_sympy(expression.left)
            right = self._to_sympy(expression.right)
            if expression.operator is BinaryOperator.ADD:
                return left + right
            if expression.operator is BinaryOperator.SUBTRACT:
                return left - right
            if expression.operator is BinaryOperator.MULTIPLY:
                return left * right
            if expression.operator is BinaryOperator.DIVIDE:
                return left / right
            if expression.operator is BinaryOperator.POWER:
                return left**right
            assert_never(expression.operator)
        if isinstance(expression, FunctionCall):
            return self._function_to_sympy(expression)
        if isinstance(expression, Relation):
            left = self._to_sympy(expression.left)
            right = self._to_sympy(expression.right)
            if expression.operator is RelationOperator.EQUAL:
                return sp.Eq(left, right, evaluate=False)
            if expression.operator is RelationOperator.NOT_EQUAL:
                return sp.Ne(left, right, evaluate=False)
            relations = {
                RelationOperator.LESS: sp.Lt,
                RelationOperator.LESS_EQUAL: sp.Le,
                RelationOperator.GREATER: sp.Gt,
                RelationOperator.GREATER_EQUAL: sp.Ge,
            }
            return relations[expression.operator](left, right)
        if isinstance(expression, SequenceExpression):
            raise BackendError("a sequence is only valid where an operation expects one")
        raise BackendError("opaque expressions cannot be sent back to the symbolic backend")

    def _function_to_sympy(self, expression: FunctionCall) -> sp.Basic:
        arguments = tuple(self._to_sympy(argument) for argument in expression.arguments)
        name = str(expression.name)
        unary_functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "asinh": sp.asinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "factorial": sp.factorial,
            "gamma": sp.gamma,
        }
        unary = unary_functions.get(name)
        if unary is not None:
            if len(arguments) != 1:
                raise QueryError(f"{name} expects one argument")
            return unary(arguments[0])
        if name == "log":
            if len(arguments) not in {1, 2}:
                raise QueryError("log expects one or two arguments")
            if len(arguments) == 1:
                return sp.log(arguments[0])
            return sp.log(arguments[0], arguments[1])
        if name == Operation.DIFFERENTIATE.value:
            if len(arguments) not in {2, 3}:
                raise QueryError("nested diff expects two or three arguments")
            order = 1
            if len(expression.arguments) == 3:
                order = _expect_integer(expression.arguments[2], role="derivative order")
            return sp.diff(arguments[0], arguments[1], order)
        return sp.Function(name)(*arguments)

    def _equations(self, expression: Expression) -> sp.Basic | Sequence[sp.Basic]:
        if isinstance(expression, SequenceExpression):
            return tuple(self._to_sympy(item) for item in expression.items)
        return self._to_sympy(expression)

    def _symbols(self, expression: Expression) -> tuple[sp.Basic, ...]:
        if isinstance(expression, SequenceExpression):
            return tuple(
                self._to_sympy(_expect_symbol(item, role="solution variable"))
                for item in expression.items
            )
        return (self._to_sympy(_expect_symbol(expression, role="solution variable")),)

    def _matrix_from_expression(self, expression: Expression) -> sp.MatrixBase:
        matrix_expression = expression
        if isinstance(expression, FunctionCall) and expression.name == "matrix":
            if len(expression.arguments) != 1:
                raise QueryError("matrix expects one nested sequence")
            matrix_expression = expression.arguments[0]
        if not isinstance(matrix_expression, SequenceExpression):
            raise QueryError("matrix data must be a nested sequence")
        rows: list[list[sp.Basic]] = []
        width: int | None = None
        for row_expression in matrix_expression.items:
            if not isinstance(row_expression, SequenceExpression):
                raise QueryError("each matrix row must be a sequence")
            row = [self._to_sympy(item) for item in row_expression.items]
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise QueryError("matrix rows must have equal length")
            rows.append(row)
        if not rows or width == 0:
            raise QueryError("matrix cannot be empty")
        return sp.Matrix(rows)

    def _to_value(self, value: object) -> MathValue:
        if isinstance(value, bool):
            return BooleanValue(value=value)
        if isinstance(value, int):
            return ScalarValue(expression=Number(value=Fraction(value)))
        if isinstance(value, sp.MatrixBase):
            rows = tuple(
                tuple(self._from_sympy(value[row, column]) for column in range(value.cols))
                for row in range(value.rows)
            )
            return MatrixValue(rows=rows)
        if isinstance(value, sp.Basic):
            return ScalarValue(expression=self._from_sympy(value))
        if _is_object_mapping(value):
            entries: list[MappingEntry] = []
            for key, item in value.items():
                key_expression = self._object_to_expression(key)
                entries.append(MappingEntry(key=key_expression, value=self._to_value(item)))
            entries.sort(key=lambda entry: str(entry.key))
            return MappingValue(entries=tuple(entries))
        if _is_object_sequence(value):
            return SequenceValue(items=tuple(self._to_value(item) for item in value))
        raise BackendError(f"unsupported backend result type: {type(value).__name__}")

    def _object_to_expression(self, value: object) -> Expression:
        if isinstance(value, int):
            return Number(value=Fraction(value))
        if isinstance(value, sp.Basic):
            return self._from_sympy(value)
        raise BackendError(f"unsupported mapping key type: {type(value).__name__}")

    def _from_sympy(self, value: sp.Basic) -> Expression:
        integration_constant = sp.Symbol("C")
        if value.func == sp.Abs and len(value.args) == 1:
            return FunctionCall(
                name=Identifier("abs"),
                arguments=(self._from_sympy(value.args[0]),),
            )
        if value.func == sp.Add and value.has(integration_constant):
            nonconstant_terms = tuple(term for term in value.args if term != integration_constant)
            if nonconstant_terms and all(
                not term.has(integration_constant) for term in nonconstant_terms
            ):
                expression = self._from_sympy(nonconstant_terms[0])
                for term in nonconstant_terms[1:]:
                    expression = BinaryExpression(
                        operator=BinaryOperator.ADD,
                        left=expression,
                        right=self._from_sympy(term),
                    )
                return BinaryExpression(
                    operator=BinaryOperator.ADD,
                    left=expression,
                    right=Symbol(name=Identifier("C")),
                )
        text = re.sub(r"\bI\b", "i", str(value)).replace("**", "^")
        if (
            _INTEGER_PATTERN.fullmatch(text) is not None
            or _RATIONAL_PATTERN.fullmatch(text) is not None
        ):
            return Number(value=Fraction(text))
        if _DECIMAL_PATTERN.fullmatch(text) is not None:
            return ApproximateNumber(text=text)
        try:
            return parse_expression(text)
        except ParseError:
            return OpaqueExpression(text=text)

    def _step_expression(self, value: object) -> Expression:
        if isinstance(value, sp.Basic):
            return self._from_sympy(value)
        if isinstance(value, bool):
            return OpaqueExpression(text="true" if value else "false")
        if isinstance(value, int):
            return Number(value=Fraction(value))
        return OpaqueExpression(text=str(value).replace("**", "^"))

    def _verify_result(self, query: Query, backend_value: object) -> Verification:
        operation = query.operation
        if operation in {
            Operation.SIMPLIFY,
            Operation.EXPAND,
            Operation.FACTOR,
            Operation.CANCEL,
            Operation.APART,
        }:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("algebraic verification requires a scalar result")
            difference = sp.simplify(self._to_sympy(query.arguments[0]) - backend_value)
            if str(difference) != "0":
                raise BackendError("the algebraic result failed symbolic equivalence verification")
            return Verification(
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="Simplifying the difference between both expressions produced zero.",
            )
        if operation is Operation.DIFFERENTIATE:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("derivative verification requires a scalar result")
            order = (
                _expect_integer(query.arguments[2], role="derivative order")
                if len(query.arguments) == 3
                else 1
            )
            expected = sp.diff(
                self._to_sympy(query.arguments[0]),
                self._to_sympy(query.arguments[1]),
                order,
            )
            if str(sp.simplify(expected - backend_value)) != "0":
                raise BackendError("the derivative result failed symbolic verification")
            return Verification(
                method=VerificationMethod.DIFFERENTIATION,
                detail="Recomputing the derivative and simplifying the difference produced zero.",
            )
        if operation is Operation.INTEGRATE and len(query.arguments) == 2:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("antiderivative verification requires a scalar result")
            derivative = sp.diff(backend_value, self._to_sympy(query.arguments[1]))
            difference = sp.simplify(derivative - self._to_sympy(query.arguments[0]))
            if str(difference) != "0":
                raise BackendError("the antiderivative failed differentiation verification")
            return Verification(
                method=VerificationMethod.DIFFERENTIATION,
                detail="Differentiating the result recovered the original integrand.",
            )
        if operation in {
            Operation.GCD,
            Operation.LCM,
            Operation.IS_PRIME,
            Operation.PRIME_FACTORS,
            Operation.BINOMIAL,
            Operation.PERMUTATIONS,
            Operation.COMBINATIONS,
        }:
            return Verification(
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The operation was evaluated using exact integer arithmetic.",
            )
        return Verification(
            method=VerificationMethod.BACKEND_IDENTITY,
            detail="The exact operation completed and crossed the validated typed boundary.",
        )
