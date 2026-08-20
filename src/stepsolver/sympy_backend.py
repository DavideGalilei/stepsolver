"""SymPy implementation of the StepSolver symbolic backend."""

import sympy as sp

from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Constant,
    ConstantName,
    FunctionCall,
    Identifier,
    Operation,
    Query,
    Relation,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.derivation.sums import find_undefined_summation
from stepsolver.errors import BackendError, QueryError
from stepsolver.results import (
    DivergenceKind,
    DivergentResult,
    ExactResult,
    NoSolutionValue,
    SolutionStep,
    SolveResult,
    UndefinedResult,
    UnsolvedResult,
    Verification,
    VerificationMethod,
)
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_execution import SympyExecutor
from stepsolver.sympy_rendering import SympyDerivationRenderer
from stepsolver.sympy_steps import SympyStepBuilder
from stepsolver.sympy_support import (
    contains_unevaluated_operation,
    expect_arity,
    expect_symbol,
    is_object_mapping,
    is_object_sequence,
    is_real_expression,
    query_expression,
    substitute,
)
from stepsolver.sympy_verification import SympyVerifier

_EQUATION_ARITY = 2
_DEFINITE_INTEGRAL_ARITY = 4
_BOUNDED_OPERATION_ARITY = 4


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
        undefined_sum = self._undefined_sum_result(query)
        if undefined_sum is not None:
            return undefined_sum
        backend_value = self._executor.execute(query)
        backend_value = self._real_solve_result(query, backend_value)
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
        after = self._converter.step_expression(backend_value)
        step = SolutionStep(
            rule="Compute exact result",
            before=query_expression(query),
            after=after,
            explanation=(
                f"Evaluate the {query.operation.value} operation with the symbolic backend."
            ),
            verification=self._verifier.verify_result(query, backend_value),
        )
        return ExactResult(query=query, value=value, steps=(step,))

    @staticmethod
    def _real_solve_result(query: Query, backend_value: object) -> object:
        if query.operation is not Operation.SOLVE or not is_object_sequence(backend_value):
            return backend_value
        filtered: list[object] = []
        for solution in backend_value:
            if not is_object_mapping(solution):
                return backend_value
            values = tuple(solution.values())
            if all(
                not isinstance(value, sp.Basic) or is_real_expression(value) for value in values
            ):
                filtered.append(solution)
        return filtered

    @staticmethod
    def _divergence_reasons(operation: Operation) -> dict[DivergenceKind, str]:
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

    def _integral_domain_reason(self, query: Query) -> str | None:
        if query.operation is not Operation.INTEGRATE or len(query.arguments) != _EQUATION_ARITY:
            return None
        variable_node = query.arguments[1]
        if not isinstance(variable_node, Symbol):
            return None
        integrand = self._converter.to_sympy(query.arguments[0])
        variable = self._converter.to_sympy(variable_node)
        if not isinstance(variable, sp.Symbol) or not integrand.is_rational_function(variable):
            return None
        _numerator, denominator = sp.fraction(sp.together(integrand))
        try:
            denominator_polynomial = sp.Poly(denominator, variable)
        except sp.PolynomialError:
            return None
        roots = sp.solve(denominator_polynomial.as_expr(), variable)
        if not is_object_sequence(roots):
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
        if query.operation is not Operation.SOLVE or len(query.arguments) != _EQUATION_ARITY:
            return None
        equation, variable = query.arguments
        if not isinstance(equation, Relation) or not isinstance(variable, Symbol):
            return None
        difference = sp.simplify(
            self._converter.to_sympy(equation.left) - self._converter.to_sympy(equation.right)
        )
        if difference != sp.Integer(0):
            return None
        return (
            "This equation is true for every value in its domain. The current result model "
            "cannot yet represent a universal solution set together with possible domain "
            "exclusions."
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
        backend_result = self._executor.execute(transformed_query)
        non_exact_reason = self._non_exact_reason(backend_result)
        if non_exact_reason is not None:
            return UnsolvedResult(query=query, reason=non_exact_reason, steps=())
        value = self._converter.to_value(backend_result)
        transformation = FunctionCall(
            name=Identifier(Operation.INTEGRATE.value),
            arguments=(transformed_integrand, parameter, lower, upper),
        )
        steps = (
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
        return ExactResult(query=query, value=value, steps=steps)
