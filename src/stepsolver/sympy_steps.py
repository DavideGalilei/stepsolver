"""Select and build detailed student-facing derivations."""

from fractions import Fraction
from typing import cast

import sympy as sp

from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    Number,
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
    Symbol,
)
from stepsolver.derivation.algebra import (
    describe_algebra_step,
    is_algebra_scalar,
    symbolic_denominators,
)
from stepsolver.derivation.definite_integrals import (
    derive_definite_integral,
    derive_dirichlet_integral,
    derive_improper_integral,
)
from stepsolver.derivation.derivatives import derive_derivative
from stepsolver.derivation.equations import derive_polynomial_equation
from stepsolver.derivation.integrals_benchmark import derive_advanced_substitution_integral
from stepsolver.derivation.integrals_elementary import (
    derive_basic_antiderivative,
    derive_constant_multiple_integral,
    derive_function_substitution_integral,
    derive_log_derivative_integral,
    derive_polynomial_sum_integral,
)
from stepsolver.derivation.integrals_special import (
    derive_gaussian_antiderivative,
    derive_integration_by_parts,
    derive_inverse_hyperbolic_integral,
    derive_partial_fraction_integral,
    derive_shifted_semicircle_integral,
    derive_square_root_rational_integral,
    derive_trigonometric_power_integral,
)
from stepsolver.derivation.integrals_structural import (
    derive_cyclic_exponential_trig_integral,
    derive_inverse_function_by_parts,
    derive_inverse_tangent_substitution,
    derive_trig_power_substitution,
)
from stepsolver.derivation.limits import derive_limit
from stepsolver.derivation.model import BackendDerivationStep
from stepsolver.derivation.reciprocal_quadratic import derive_reciprocal_quadratic_integral
from stepsolver.derivation.sums import derive_sum
from stepsolver.derivation.systems import derive_linear_system
from stepsolver.derivation.transcendental_equations import derive_transcendental_equation
from stepsolver.results import (
    SolutionStep,
    StepConstraint,
    Verification,
    VerificationMethod,
)
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_rendering import SympyDerivationRenderer
from stepsolver.sympy_support import is_object_mapping, is_object_sequence

_MIN_DERIVATIVE_ARGUMENTS = 2
_MAX_DERIVATIVE_ARGUMENTS = 3


def _derivative_order(query: Query) -> int | None:
    if len(query.arguments) == _MIN_DERIVATIVE_ARGUMENTS:
        return 1
    order_expression = query.arguments[2]
    if not isinstance(order_expression, Number):
        return None
    order = order_expression.value
    if order.denominator != 1 or order <= 0:
        return None
    return order.numerator


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
        derive_trig_power_substitution,
        derive_inverse_tangent_substitution,
        derive_square_root_rational_integral,
        derive_shifted_semicircle_integral,
        derive_inverse_hyperbolic_integral,
        derive_trigonometric_power_integral,
        derive_inverse_function_by_parts,
        derive_cyclic_exponential_trig_integral,
        derive_integration_by_parts,
        derive_advanced_substitution_integral,
        derive_gaussian_antiderivative,
        derive_reciprocal_quadratic_integral,
        derive_partial_fraction_integral,
    )
    for strategy in strategies:
        derivation = strategy(integrand, variable, result)
        if derivation:
            return derivation
    return ()


def _domain_denominators(expression: Expression) -> tuple[Expression, ...]:
    """Collect denominator expressions whose zeros are outside the input domain."""
    if isinstance(expression, BinaryExpression):
        nested = _domain_denominators(expression.left) + _domain_denominators(expression.right)
        if expression.operator is BinaryOperator.DIVIDE:
            return (expression.right, *nested)
        if (
            expression.operator is BinaryOperator.POWER
            and isinstance(expression.right, Number)
            and expression.right.value < 0
        ):
            return (expression.left, *nested)
        return nested
    if isinstance(expression, Relation):
        return _domain_denominators(expression.left) + _domain_denominators(expression.right)
    if isinstance(expression, SequenceExpression):
        return tuple(
            denominator for item in expression.items for denominator in _domain_denominators(item)
        )
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


class SympyStepBuilder:
    """Choose a derivation strategy and convert it to public solution steps."""

    def __init__(self, converter: SympyConverter, renderer: SympyDerivationRenderer) -> None:
        """Use shared conversion and rendering collaborators."""
        self._converter = converter
        self._renderer = renderer

    def detailed_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Choose detailed steps for a supported operation family."""
        match query.operation:
            case (
                Operation.SIMPLIFY
                | Operation.EXPAND
                | Operation.FACTOR
                | Operation.CANCEL
                | Operation.APART
            ):
                steps = self.detailed_algebra_steps(query, backend_value)
            case Operation.SOLVE:
                steps = self.detailed_equation_steps(query, backend_value)
            case Operation.DIFFERENTIATE:
                steps = self.detailed_derivative_steps(query, backend_value)
            case Operation.INTEGRATE:
                steps = self.detailed_integral_steps(query, backend_value)
            case Operation.LIMIT:
                steps = self.detailed_limit_steps(query, backend_value)
            case Operation.SUM:
                steps = self.detailed_sum_steps(query, backend_value)
            case _:
                steps = ()
        return steps

    def detailed_algebra_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Describe elementary algebra using the learner's original expression."""
        if len(query.arguments) not in {1, 2} or not isinstance(backend_value, sp.Basic):
            return ()
        original = query.arguments[0]
        if not is_algebra_scalar(original):
            return ()
        backend_expression = self._converter.to_sympy(original)
        if sp.simplify(backend_expression - backend_value) != sp.Integer(0):
            return ()
        description = describe_algebra_step(
            query.operation,
            original,
            backend_expression,
            backend_value,
        )
        denominators = (
            symbolic_denominators(original)
            if description.rule == "Cancel common factors"
            else ()
        )
        constraints = tuple(
            StepConstraint(
                explanation="An original denominator cannot equal zero.",
                expression=Relation(
                    operator=RelationOperator.NOT_EQUAL,
                    left=denominator,
                    right=Number(value=Fraction(0)),
                ),
            )
            for denominator in denominators
        )
        return (
            SolutionStep(
                rule=description.rule,
                before=original,
                after=self._converter.step_expression(backend_value),
                explanation=description.explanation,
                verification=Verification(
                    method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    detail=(
                        "The expressions agree wherever the original denominators are nonzero."
                        if constraints
                        else "Simplifying the difference between both expressions gives zero."
                    ),
                ),
                introduced_constraints=constraints,
            ),
        )

    def detailed_sum_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build human-readable steps for common finite and infinite sums."""
        if len(query.arguments) != 4 or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node, lower_node, upper_node = query.arguments
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        lower = self._converter.to_sympy(lower_node)
        upper = self._converter.to_sympy(upper_node)
        if not isinstance(variable, sp.Symbol):
            return ()
        try:
            derivation = derive_sum(expression, variable, lower, upper, backend_value)
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_limit_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build detailed steps for a limit query when a strategy matches."""
        if len(query.arguments) not in {3, 4} or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node, point_node = query.arguments[:3]
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        point = self._converter.to_sympy(point_node)
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
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_equation_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build detailed steps for a polynomial equation when possible."""
        if len(query.arguments) != 2:
            return ()
        equation_expression, variable_expression = query.arguments
        if isinstance(equation_expression, SequenceExpression) and isinstance(
            variable_expression, SequenceExpression
        ):
            return self.detailed_system_steps(
                equation_expression,
                variable_expression,
                backend_value,
            )
        if not isinstance(equation_expression, Relation) or not isinstance(
            variable_expression,
            Symbol,
        ):
            return ()
        equation = self._converter.to_sympy(equation_expression)
        variable = self._converter.to_sympy(variable_expression)
        if not isinstance(equation, sp.Equality) or not isinstance(variable, sp.Symbol):
            return ()
        roots = self.solution_roots(backend_value, variable)
        try:
            derivation = derive_transcendental_equation(
                equation,
                variable,
                roots,
            )
            if not derivation:
                domain_denominators = tuple(
                    self._converter.to_sympy(item)
                    for item in _domain_denominators(equation_expression)
                )
                derivation = derive_polynomial_equation(
                    equation,
                    variable,
                    roots,
                    domain_denominators,
                )
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_system_steps(
        self,
        equations: SequenceExpression,
        variables: SequenceExpression,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build elimination and back-substitution steps for a linear system."""
        if len(equations.items) < 2 or len(variables.items) < 2:
            return ()
        if not all(isinstance(item, Relation) for item in equations.items) or not all(
            isinstance(item, Symbol) for item in variables.items
        ):
            return ()
        system_equations = tuple(self._converter.to_sympy(item) for item in equations.items)
        system_variables = tuple(self._converter.to_sympy(item) for item in variables.items)
        if not all(isinstance(item, sp.Equality) for item in system_equations):
            return ()
        if not all(isinstance(item, sp.Symbol) for item in system_variables):
            return ()
        derivation = derive_linear_system(
            cast("tuple[sp.Equality, ...]", system_equations),
            cast("tuple[sp.Symbol, ...]", system_variables),
            backend_value,
        )
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_derivative_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build detailed steps for a supported derivative rule."""
        if len(query.arguments) not in {
            _MIN_DERIVATIVE_ARGUMENTS,
            _MAX_DERIVATIVE_ARGUMENTS,
        } or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node = query.arguments[:2]
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        order = _derivative_order(query)
        if not isinstance(variable, sp.Symbol) or order is None:
            return ()
        try:
            derivation: tuple[BackendDerivationStep, ...] = ()
            current = expression
            for _index in range(order):
                differentiated = sp.diff(current, variable)
                derivation = (*derivation, *derive_derivative(current, variable, differentiated))
                current = differentiated
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        if sp.simplify(current - backend_value) != sp.Integer(0):
            return ()
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_integral_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build detailed steps for a supported integral strategy."""
        if len(query.arguments) not in {2, 4} or not isinstance(backend_value, sp.Basic):
            return ()
        integrand_expression, variable_expression = query.arguments[:2]
        if not isinstance(variable_expression, Symbol):
            return ()
        integrand = self._converter.to_sympy(integrand_expression)
        variable = self._converter.to_sympy(variable_expression)
        if not isinstance(variable, sp.Symbol):
            return ()
        if len(query.arguments) == 2:
            try:
                derivation = _derive_indefinite_integral(integrand, variable, backend_value)
            except (sp.PolynomialError, TypeError, ValueError):
                return ()
        else:
            lower = self._converter.to_sympy(query.arguments[2])
            upper = self._converter.to_sympy(query.arguments[3])
            try:
                derivation = _derive_definite_integral(
                    integrand, variable, lower, upper, backend_value
                )
            except (sp.PolynomialError, TypeError, ValueError):
                return ()
        return tuple(self._renderer.solution_step(item) for item in derivation)

    @staticmethod
    def solution_roots(
        backend_value: object,
        variable: sp.Symbol,
    ) -> tuple[sp.Basic, ...]:
        """Extract one target symbol's roots from SymPy solution mappings."""
        if not is_object_sequence(backend_value):
            return ()
        roots: list[sp.Basic] = []
        for solution in backend_value:
            if not is_object_mapping(solution):
                return ()
            root = solution.get(variable)
            if not isinstance(root, sp.Basic):
                return ()
            roots.append(root)
        return tuple(roots)
