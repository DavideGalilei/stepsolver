"""Select and build detailed student-facing derivations."""

import sympy as sp

from stepsolver.ast import (
    Operation,
    Query,
    Relation,
    Symbol,
)
from stepsolver.derivation.definite_integrals import (
    derive_definite_integral,
    derive_dirichlet_integral,
    derive_improper_integral,
)
from stepsolver.derivation.derivatives import derive_derivative
from stepsolver.derivation.equations import derive_polynomial_equation
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
from stepsolver.derivation.limits import derive_limit
from stepsolver.derivation.model import BackendDerivationStep
from stepsolver.derivation.reciprocal_quadratic import derive_reciprocal_quadratic_integral
from stepsolver.results import (
    SolutionStep,
)
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_rendering import SympyDerivationRenderer
from stepsolver.sympy_support import is_object_mapping, is_object_sequence


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
        if query.operation is Operation.SOLVE:
            return self.detailed_equation_steps(query, backend_value)
        if query.operation is Operation.DIFFERENTIATE:
            return self.detailed_derivative_steps(query, backend_value)
        if query.operation is Operation.INTEGRATE:
            return self.detailed_integral_steps(query, backend_value)
        if query.operation is Operation.LIMIT:
            return self.detailed_limit_steps(query, backend_value)
        return ()

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
            derivation = derive_polynomial_equation(equation, variable, roots)
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._renderer.solution_step(item) for item in derivation)

    def detailed_derivative_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        """Build detailed steps for a supported derivative rule."""
        if len(query.arguments) != 2 or not isinstance(backend_value, sp.Basic):
            return ()
        expression_node, variable_node = query.arguments
        if not isinstance(variable_node, Symbol):
            return ()
        expression = self._converter.to_sympy(expression_node)
        variable = self._converter.to_sympy(variable_node)
        if not isinstance(variable, sp.Symbol):
            return ()
        try:
            derivation = derive_derivative(expression, variable, backend_value)
        except (sp.PolynomialError, TypeError, ValueError):
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
