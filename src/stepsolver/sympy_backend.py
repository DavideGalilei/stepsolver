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
    BackendDifferential,
    BackendExpression,
    BackendIdentity,
    BackendIntegral,
    derive_polynomial_equation,
    derive_reciprocal_quadratic_integral,
)

_INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")
_RATIONAL_PATTERN = re.compile(r"^-?[0-9]+/[0-9]+$")
_DECIMAL_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]+$")


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _query_expression(query: Query) -> FunctionCall:
    return FunctionCall(name=Identifier(query.operation.value), arguments=query.arguments)


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
        backend_value = self._execute(query)
        value = self._to_value(backend_value)
        detailed_steps = self._detailed_steps(query, backend_value)
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

    def _detailed_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if query.operation is Operation.SOLVE:
            return self._detailed_equation_steps(query, backend_value)
        if query.operation is Operation.INTEGRATE:
            return self._detailed_integral_steps(query, backend_value)
        return ()

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
        if not roots:
            return ()
        try:
            derivation = derive_polynomial_equation(equation, variable, roots)
        except (sp.PolynomialError, TypeError, ValueError):
            return ()
        return tuple(self._solution_step(item) for item in derivation)

    def _detailed_integral_steps(
        self,
        query: Query,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if len(query.arguments) != 2 or not isinstance(backend_value, sp.Basic):
            return ()
        integrand_expression, variable_expression = query.arguments
        if not isinstance(variable_expression, Symbol):
            return ()
        integrand = self._to_sympy(integrand_expression)
        variable = self._to_sympy(variable_expression)
        if not isinstance(variable, sp.Symbol):
            return ()
        try:
            derivation = derive_reciprocal_quadratic_integral(
                integrand,
                variable,
                backend_value,
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
        if isinstance(value, BackendDifferential):
            return FunctionCall(
                name=Identifier("differential"),
                arguments=(self._from_sympy(value.variable),),
            )
        if isinstance(value, BackendIntegral):
            return FunctionCall(
                name=Identifier(Operation.INTEGRATE.value),
                arguments=(
                    self._from_sympy(value.integrand),
                    self._from_sympy(value.variable),
                ),
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
                return sp.factor(
                    sp.diff(
                        self._to_sympy(arguments[0]),
                        self._to_sympy(arguments[1]),
                        order,
                    )
                )
            case Operation.INTEGRATE:
                _expect_arity(query, 2, 4)
                integrand = self._to_sympy(arguments[0])
                variable = self._to_sympy(arguments[1])
                if len(arguments) == 2:
                    return sp.integrate(integrand, variable) + sp.Symbol("C")
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
            relations = {
                RelationOperator.EQUAL: sp.Eq,
                RelationOperator.NOT_EQUAL: sp.Ne,
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
            return sp.log(arguments[0], arguments[1] if len(arguments) == 2 else None)
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
