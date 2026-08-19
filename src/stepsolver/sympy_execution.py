"""Dispatch StepSolver operations to SymPy."""

import sympy as sp

from stepsolver.ast import (
    Operation,
    Query,
    Symbol,
)
from stepsolver.errors import BackendError, QueryError
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_series import match_alternating_p_series, match_harmonic_sine_series
from stepsolver.sympy_support import expect_arity, expect_integer, expect_symbol


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


class SympyExecutor:
    """Execute validated queries without handling presentation or pedagogy."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the supplied converter for query arguments and structured values."""
        self._converter = converter

    def _evaluate_sum(self, query: Query) -> sp.Basic:
        """Evaluate a sum, including exact series families SymPy leaves inert."""
        expect_arity(query, 4)
        expression = self._converter.to_sympy(query.arguments[0])
        variable = self._converter.to_sympy(query.arguments[1])
        lower = self._converter.to_sympy(query.arguments[2])
        upper = self._converter.to_sympy(query.arguments[3])
        if isinstance(variable, sp.Symbol):
            harmonic_sine = match_harmonic_sine_series(
                expression,
                variable,
                lower,
                upper,
            )
            if harmonic_sine is not None:
                return harmonic_sine.value
            alternating = match_alternating_p_series(expression, variable, lower, upper)
            if alternating is not None:
                return alternating.value
        return sp.summation(expression, (variable, lower, upper))

    def execute(self, query: Query) -> object:
        """Execute one validated query and return its backend-native result."""
        operation = query.operation
        arguments = query.arguments
        match operation:
            case Operation.SIMPLIFY:
                expect_arity(query, 1)
                return sp.simplify(self._converter.to_sympy(arguments[0]))
            case Operation.EXPAND:
                expect_arity(query, 1)
                return sp.expand(self._converter.to_sympy(arguments[0]))
            case Operation.FACTOR:
                expect_arity(query, 1)
                return sp.factor(self._converter.to_sympy(arguments[0]))
            case Operation.CANCEL:
                expect_arity(query, 1)
                return sp.cancel(self._converter.to_sympy(arguments[0]))
            case Operation.APART:
                expect_arity(query, 1, 2)
                variable = self._converter.to_sympy(arguments[1]) if len(arguments) == 2 else None
                return sp.apart(self._converter.to_sympy(arguments[0]), variable)
            case Operation.SOLVE:
                expect_arity(query, 2)
                equations = self._converter.equations(arguments[0])
                symbols = self._converter.symbols(arguments[1])
                return sp.solve(equations, *symbols, dict=True)
            case Operation.SOLVE_INEQUALITY:
                expect_arity(query, 2)
                symbol = expect_symbol(arguments[1], role="inequality variable")
                backend_symbol = self._converter.to_sympy(symbol)
                if not isinstance(backend_symbol, sp.Symbol):
                    raise BackendError("symbol conversion did not produce a SymPy Symbol")
                return sp.solve_univariate_inequality(
                    self._converter.to_sympy(arguments[0]), backend_symbol
                )
            case Operation.DIFFERENTIATE:
                expect_arity(query, 2, 3)
                order = (
                    expect_integer(arguments[2], role="derivative order")
                    if len(arguments) == 3
                    else 1
                )
                return sp.diff(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    order,
                )
            case Operation.INTEGRATE:
                expect_arity(query, 2, 4)
                integrand = self._converter.to_sympy(arguments[0])
                variable = self._converter.to_sympy(arguments[1])
                if len(arguments) == 2:
                    return _evaluate_indefinite_integral(integrand, variable)
                return sp.integrate(
                    integrand,
                    (
                        variable,
                        self._converter.to_sympy(arguments[2]),
                        self._converter.to_sympy(arguments[3]),
                    ),
                )
            case Operation.CONTOUR_INTEGRATE:
                raise BackendError("contour integrals require the dedicated execution path")
            case Operation.LIMIT:
                expect_arity(query, 3, 4)
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
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    direction,
                )
            case Operation.SERIES:
                expect_arity(query, 4)
                return sp.series(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    expect_integer(arguments[3], role="series order"),
                )
            case Operation.SUM:
                return self._evaluate_sum(query)
            case Operation.PRODUCT:
                expect_arity(query, 4)
                return sp.product(
                    self._converter.to_sympy(arguments[0]),
                    (
                        self._converter.to_sympy(arguments[1]),
                        self._converter.to_sympy(arguments[2]),
                        self._converter.to_sympy(arguments[3]),
                    ),
                )
            case Operation.MATRIX:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0])
            case Operation.DETERMINANT:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).det()
            case Operation.INVERSE:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).inv()
            case Operation.RANK:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).rank()
            case Operation.RREF:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).rref()
            case Operation.EIGENVALUES:
                expect_arity(query, 1)
                return self._converter.matrix_from_expression(arguments[0]).eigenvals()
            case Operation.DSOLVE:
                expect_arity(query, 2)
                return sp.dsolve(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.RSOLVE:
                expect_arity(query, 2)
                return sp.rsolve(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.LAPLACE:
                expect_arity(query, 3)
                return sp.laplace_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                    noconds=True,
                )
            case Operation.INVERSE_LAPLACE:
                expect_arity(query, 3)
                return sp.inverse_laplace_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.FOURIER:
                expect_arity(query, 3)
                return sp.fourier_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.INVERSE_FOURIER:
                expect_arity(query, 3)
                return sp.inverse_fourier_transform(
                    self._converter.to_sympy(arguments[0]),
                    self._converter.to_sympy(arguments[1]),
                    self._converter.to_sympy(arguments[2]),
                )
            case Operation.GCD:
                expect_arity(query, 2)
                return sp.gcd(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.LCM:
                expect_arity(query, 2)
                return sp.lcm(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.IS_PRIME:
                expect_arity(query, 1)
                return sp.isprime(expect_integer(arguments[0], role="primality input"))
            case Operation.PRIME_FACTORS:
                expect_arity(query, 1)
                return sp.factorint(expect_integer(arguments[0], role="factorization input"))
            case Operation.BINOMIAL | Operation.COMBINATIONS:
                expect_arity(query, 2)
                return sp.binomial(
                    self._converter.to_sympy(arguments[0]), self._converter.to_sympy(arguments[1])
                )
            case Operation.PERMUTATIONS:
                expect_arity(query, 2)
                n = self._converter.to_sympy(arguments[0])
                r = self._converter.to_sympy(arguments[1])
                return sp.factorial(n) / sp.factorial(n - r)
            case Operation.NUMERIC:
                expect_arity(query, 1, 2)
                digits = (
                    expect_integer(arguments[1], role="precision") if len(arguments) == 2 else 15
                )
                return sp.N(self._converter.to_sympy(arguments[0]), digits)
