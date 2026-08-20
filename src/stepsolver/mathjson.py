"""Convert validated MathJSON notation into StepSolver AST queries."""

from fractions import Fraction

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
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.errors import QueryError
from stepsolver.formatter import format_expression

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None

_CONSTANTS: dict[str, ConstantName] = {
    "Pi": ConstantName.PI,
    "ExponentialE": ConstantName.E,
    "e": ConstantName.E,
    "ImaginaryUnit": ConstantName.IMAGINARY,
    "PositiveInfinity": ConstantName.INFINITY,
    "Infinity": ConstantName.INFINITY,
}
_FUNCTIONS: dict[str, str] = {
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Arcsin": "asin",
    "Arccos": "acos",
    "Arctan": "atan",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "Tanh": "tanh",
    "Exp": "exp",
    "Ln": "log",
    "Log": "log",
    "Sqrt": "sqrt",
    "Root": "root",
    "Abs": "abs",
    "Gamma": "gamma",
    "Factorial": "factorial",
}
_RELATIONS: dict[str, RelationOperator] = {
    "Equal": RelationOperator.EQUAL,
    "NotEqual": RelationOperator.NOT_EQUAL,
    "Less": RelationOperator.LESS,
    "LessEqual": RelationOperator.LESS_EQUAL,
    "Greater": RelationOperator.GREATER,
    "GreaterEqual": RelationOperator.GREATER_EQUAL,
}


def _error(message: str) -> QueryError:
    return QueryError(message)


def _node(value: JsonValue, *, role: str) -> list[JsonValue]:
    if not isinstance(value, list) or not value:
        raise _error(f"{role} must be a MathJSON function expression")
    return value


def _head(node: list[JsonValue]) -> str:
    head = node[0]
    if not isinstance(head, str):
        raise _error("MathJSON function names must be strings")
    return head


def _exact_arity(node: list[JsonValue], count: int, *, name: str) -> None:
    if len(node) != count + 1:
        raise _error(f"{name} expects {count} argument(s)")


def _fold_binary(operator: BinaryOperator, items: list[Expression]) -> Expression:
    if not items:
        raise _error(f"{operator.value} requires at least one operand")
    result = items[0]
    for item in items[1:]:
        result = BinaryExpression(operator=operator, left=result, right=item)
    return result


def expression_from_mathjson(value: JsonValue) -> Expression:
    """Decode one MathJSON expression into StepSolver's closed AST."""
    if value is None or isinstance(value, bool | dict):
        raise _error("unsupported MathJSON literal")
    if isinstance(value, int):
        return Number(value=Fraction(value))
    if isinstance(value, float):
        return ApproximateNumber(text=str(value))
    if isinstance(value, str):
        constant = _CONSTANTS.get(value)
        if constant is not None:
            return Constant(name=constant)
        if value.startswith("'"):
            raise _error("string literals are not mathematical expressions")
        return Symbol(name=Identifier(value))

    node = _node(value, role="expression")
    head = _head(node)
    arguments = node[1:]
    if head == "Error":
        raise _error("the graphical expression is incomplete or invalid")
    if head in {"Delimiter", "Hold", "ReleaseHold"}:
        _exact_arity(node, 1, name=head)
        return expression_from_mathjson(arguments[0])
    if head == "Add":
        return _fold_binary(
            BinaryOperator.ADD,
            [expression_from_mathjson(item) for item in arguments],
        )
    if head in {"Multiply", "InvisibleOperator"}:
        return _fold_binary(
            BinaryOperator.MULTIPLY,
            [expression_from_mathjson(item) for item in arguments],
        )
    if head == "Subtract":
        _exact_arity(node, 2, name=head)
        return BinaryExpression(
            operator=BinaryOperator.SUBTRACT,
            left=expression_from_mathjson(arguments[0]),
            right=expression_from_mathjson(arguments[1]),
        )
    if head in {"Divide", "Rational"}:
        _exact_arity(node, 2, name=head)
        return BinaryExpression(
            operator=BinaryOperator.DIVIDE,
            left=expression_from_mathjson(arguments[0]),
            right=expression_from_mathjson(arguments[1]),
        )
    if head == "Power":
        _exact_arity(node, 2, name=head)
        return BinaryExpression(
            operator=BinaryOperator.POWER,
            left=expression_from_mathjson(arguments[0]),
            right=expression_from_mathjson(arguments[1]),
        )
    if head == "Square":
        _exact_arity(node, 1, name=head)
        return BinaryExpression(
            operator=BinaryOperator.POWER,
            left=expression_from_mathjson(arguments[0]),
            right=Number(value=Fraction(2)),
        )
    if head == "Negate":
        _exact_arity(node, 1, name=head)
        return UnaryExpression(
            operator=UnaryOperator.NEGATIVE,
            operand=expression_from_mathjson(arguments[0]),
        )
    relation = _RELATIONS.get(head)
    if relation is not None:
        _exact_arity(node, 2, name=head)
        return Relation(
            operator=relation,
            left=expression_from_mathjson(arguments[0]),
            right=expression_from_mathjson(arguments[1]),
        )
    if head in {"List", "Tuple"}:
        return SequenceExpression(items=tuple(expression_from_mathjson(item) for item in arguments))
    function = _FUNCTIONS.get(head)
    if function is not None:
        return FunctionCall(
            name=Identifier(function),
            arguments=tuple(expression_from_mathjson(item) for item in arguments),
        )
    if head in {"Integrate", "D", "Derivative", "Limit", "Sum", "Product"}:
        raise _error(f"{head} is only valid as the outermost operation")
    return FunctionCall(
        name=Identifier(head),
        arguments=tuple(expression_from_mathjson(item) for item in arguments),
    )


def _symbols(expression: Expression) -> frozenset[Symbol]:
    if isinstance(expression, Symbol):
        return frozenset({expression})
    if isinstance(expression, Number | ApproximateNumber | Constant):
        return frozenset()
    if isinstance(expression, UnaryExpression):
        return _symbols(expression.operand)
    if isinstance(expression, BinaryExpression | Relation):
        return _symbols(expression.left) | _symbols(expression.right)
    if isinstance(expression, FunctionCall):
        result: frozenset[Symbol] = frozenset()
        for argument in expression.arguments:
            result |= _symbols(argument)
        return result
    if isinstance(expression, SequenceExpression):
        result = frozenset()
        for item in expression.items:
            result |= _symbols(item)
        return result
    return frozenset()


def _inferred_symbol(expression: Expression, *, role: str) -> Symbol:
    symbols = sorted(_symbols(expression), key=lambda item: item.name)
    if len(symbols) != 1:
        raise _error(f"could not infer a unique {role}")
    return symbols[0]


def _query(operation: Operation, arguments: tuple[Expression, ...]) -> Query:
    displayed = FunctionCall(name=Identifier(operation.value), arguments=arguments)
    return Query(operation=operation, arguments=arguments, source=format_expression(displayed))


def _integral_query(node: list[JsonValue]) -> Query:
    if len(node) not in {2, 3}:
        raise _error("only single-variable integrals are supported in the web editor")
    integrand = expression_from_mathjson(node[1])
    variable: Expression
    if len(node) == 2:
        variable = _inferred_symbol(integrand, role="integration variable")
        return _query(Operation.INTEGRATE, (integrand, variable))
    limit_specification = node[2]
    if isinstance(limit_specification, str):
        variable = expression_from_mathjson(limit_specification)
        return _query(Operation.INTEGRATE, (integrand, variable))
    limits = _node(limit_specification, role="integration limits")
    if _head(limits) != "Tuple":
        raise _error("integration limits must be a MathJSON Tuple")
    if len(limits) == 4:
        variable = expression_from_mathjson(limits[1])
        lower = expression_from_mathjson(limits[2])
        upper = expression_from_mathjson(limits[3])
    elif len(limits) == 3:
        variable = _inferred_symbol(integrand, role="integration variable")
        lower = expression_from_mathjson(limits[1])
        upper = expression_from_mathjson(limits[2])
    else:
        raise _error("integration limits require a variable, lower bound, and upper bound")
    return _query(Operation.INTEGRATE, (integrand, variable, lower, upper))


def _derivative_query(node: list[JsonValue]) -> Query:
    variables: list[Expression] = []
    current = ["D", *node[1:]] if _head(node) == "Derivative" else node
    while _head(current) == "D":
        if len(current) < 3:
            raise _error("derivative notation requires an expression and variable")
        variables.extend(expression_from_mathjson(item) for item in current[2:])
        inner = current[1]
        if isinstance(inner, list) and inner and inner[0] == "D":
            current = inner
        else:
            expression = expression_from_mathjson(inner)
            break
    else:
        raise _error("invalid derivative notation")
    variable = variables[0]
    if any(item != variable for item in variables[1:]):
        raise _error("mixed partial derivatives are not supported yet")
    if len(variables) == 1:
        return _query(Operation.DIFFERENTIATE, (expression, variable))
    order = Number(value=Fraction(len(variables)))
    return _query(Operation.DIFFERENTIATE, (expression, variable, order))


def _limit_query(node: list[JsonValue]) -> Query:
    if len(node) == 4:
        arguments = tuple(expression_from_mathjson(item) for item in node[1:])
        return _query(Operation.LIMIT, arguments)
    if len(node) == 3:
        bound_function = node[1]
        if (
            isinstance(bound_function, list)
            and bound_function
            and _head(bound_function) == "Function"
        ):
            if len(bound_function) != 3:
                raise _error("limit function must include an expression and variable")
            return _query(
                Operation.LIMIT,
                (
                    expression_from_mathjson(bound_function[1]),
                    expression_from_mathjson(bound_function[2]),
                    expression_from_mathjson(node[2]),
                ),
            )
        limits = _node(node[2], role="limit approach")
        if _head(limits) != "Tuple" or len(limits) != 3:
            raise _error("limit notation must include a variable and approach point")
        return _query(
            Operation.LIMIT,
            (
                expression_from_mathjson(node[1]),
                expression_from_mathjson(limits[1]),
                expression_from_mathjson(limits[2]),
            ),
        )
    raise _error("limit notation must include an expression, variable, and point")


def _bounded_query(node: list[JsonValue], operation: Operation) -> Query:
    if len(node) != 3:
        raise _error(f"{operation.value} notation requires an expression and limits")
    expression = expression_from_mathjson(node[1])
    limits = _node(node[2], role=f"{operation.value} limits")
    if _head(limits) != "Tuple" or len(limits) != 4:
        raise _error(f"{operation.value} limits require a variable, lower bound, and upper bound")
    return _query(
        operation,
        (
            expression,
            expression_from_mathjson(limits[1]),
            expression_from_mathjson(limits[2]),
            expression_from_mathjson(limits[3]),
        ),
    )


def query_from_mathjson(value: JsonValue) -> Query:
    """Infer solver intent from the outer mathematical notation."""
    if isinstance(value, list) and value:
        node = value
        head = _head(node)
        if head == "Integrate":
            return _integral_query(node)
        if head in {"D", "Derivative"}:
            return _derivative_query(node)
        if head == "Limit":
            return _limit_query(node)
        if head == "Sum":
            return _bounded_query(node, Operation.SUM)
        if head == "Product":
            return _bounded_query(node, Operation.PRODUCT)
    expression = expression_from_mathjson(value)
    if (
        isinstance(expression, SequenceExpression)
        and expression.items
        and all(
            isinstance(item, Relation) and item.operator is RelationOperator.EQUAL
            for item in expression.items
        )
    ):
        symbols = tuple(sorted(_symbols(expression), key=lambda item: item.name))
        if not symbols:
            raise _error("a system must contain at least one variable")
        system_variables: Expression = (
            symbols[0] if len(symbols) == 1 else SequenceExpression(items=symbols)
        )
        return _query(Operation.SOLVE, (expression, system_variables))
    if isinstance(expression, Relation):
        symbols = tuple(sorted(_symbols(expression), key=lambda item: item.name))
        if not symbols:
            raise _error("an equation must contain at least one variable")
        variables: Expression = (
            symbols[0] if len(symbols) == 1 else SequenceExpression(items=symbols)
        )
        operation = (
            Operation.SOLVE
            if expression.operator is RelationOperator.EQUAL
            else Operation.SOLVE_INEQUALITY
        )
        return _query(operation, (expression, variables))
    if (
        isinstance(expression, SequenceExpression)
        and expression.items
        and all(isinstance(item, SequenceExpression) for item in expression.items)
    ):
        return _query(Operation.MATRIX, (expression,))
    return _query(Operation.SIMPLIFY, (expression,))
