"""LaTeX rendering for StepSolver expressions and values."""

from fractions import Fraction

from stepsolver.ast import (
    ApproximateNumber,
    BinaryExpression,
    BinaryOperator,
    Constant,
    Expression,
    FunctionCall,
    Number,
    OpaqueExpression,
    Relation,
    SequenceExpression,
    Symbol,
    UnaryExpression,
    UnaryOperator,
)
from stepsolver.results import (
    BooleanValue,
    MappingValue,
    MathValue,
    NoSolutionValue,
    ScalarValue,
    SequenceValue,
)

_PRECEDENCE: dict[BinaryOperator, int] = {
    BinaryOperator.ADD: 10,
    BinaryOperator.SUBTRACT: 10,
    BinaryOperator.MULTIPLY: 20,
    BinaryOperator.DIVIDE: 20,
    BinaryOperator.POWER: 30,
}
_GREEK_SYMBOLS: frozenset[str] = frozenset(
    {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "theta",
        "lambda",
        "mu",
        "pi",
        "rho",
        "sigma",
        "tau",
        "phi",
        "psi",
        "omega",
    }
)
_NAMED_FUNCTIONS: dict[str, str] = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "log": r"\log",
    "exp": r"\exp",
    "gamma": r"\Gamma",
    "zeta": r"\zeta",
    "erf": r"\operatorname{erf}",
    "asinh": r"\operatorname{arsinh}",
}


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def _escape_identifier(identifier: str) -> str:
    if identifier in _GREEK_SYMBOLS:
        return rf"\{identifier}"
    return identifier.replace("_", r"\_")


def _format_quadratic_solutions(arguments: tuple[Expression, ...]) -> str:
    variable, negative, positive, denominator = (
        format_latex_expression(argument) for argument in arguments
    )
    return (
        rf"\left[{variable} = \frac{{{negative}}}{{{denominator}}}, "
        rf"{variable} = \frac{{{positive}}}{{{denominator}}}\right]"
    )


def _format_evaluation_at_bounds(arguments: tuple[Expression, ...]) -> str:
    value, variable, lower, upper = (format_latex_expression(argument) for argument in arguments)
    if lower == upper:
        return rf"\left.{value}\right|_{{{variable}={lower}}}"
    return rf"\left[{value}\right]_{{{lower}}}^{{{upper}}}"


def _format_calculus_function(name: str, arguments: tuple[Expression, ...]) -> str | None:
    if name == "integrate" and len(arguments) in {2, 4}:
        integrand = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        if len(arguments) == 2:
            return rf"\int {integrand}\,\mathrm{{d}}{variable}"
        lower = format_latex_expression(arguments[2])
        upper = format_latex_expression(arguments[3])
        return rf"\int_{{{lower}}}^{{{upper}}} {integrand}\,\mathrm{{d}}{variable}"
    if name == "contour_integrate" and len(arguments) == 6:
        integrand = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        path = format_latex_expression(arguments[2])
        return rf"\int_{{{path}}} {integrand}\,\mathrm{{d}}{variable}"
    if name == "diff" and len(arguments) in {2, 3}:
        value = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        if len(arguments) == 2:
            return rf"\frac{{\mathrm{{d}}}}{{\mathrm{{d}}{variable}}}\left({value}\right)"
        order = format_latex_expression(arguments[2])
        return (
            rf"\frac{{\mathrm{{d}}^{{{order}}}}}"
            rf"{{\mathrm{{d}}{variable}^{{{order}}}}}\left({value}\right)"
        )
    if name == "limit" and len(arguments) in {3, 4}:
        value = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        point = format_latex_expression(arguments[2])
        if len(arguments) == 4 and isinstance(arguments[3], Symbol):
            direction = {"left": "-", "right": "+"}.get(arguments[3].name)
            if direction is not None:
                point = rf"{point}^{{{direction}}}"
        return rf"\lim_{{{variable} \to {point}}} {value}"
    return None


def _format_function(expression: FunctionCall) -> str:
    name = str(expression.name)
    arguments = expression.arguments
    calculus = _format_calculus_function(name, arguments)
    if calculus is not None:
        return calculus
    if name in {"sum", "product"} and len(arguments) == 4:
        value = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        lower = format_latex_expression(arguments[2])
        upper = format_latex_expression(arguments[3])
        operator = r"\sum" if name == "sum" else r"\prod"
        return rf"{operator}_{{{variable}={lower}}}^{{{upper}}} {value}"
    if name == "solve" and len(arguments) == 2:
        equation = format_latex_expression(arguments[0])
        variable = format_latex_expression(arguments[1])
        return rf"{equation}\quad\text{{solve for }}{variable}"
    if name == "det" and len(arguments) == 1:
        return rf"\det\left({format_latex_expression(arguments[0])}\right)"
    if name == "Eq" and len(arguments) == 2:
        return f"{format_latex_expression(arguments[0])} = {format_latex_expression(arguments[1])}"
    if name == "sqrt" and len(arguments) == 1:
        return rf"\sqrt{{{format_latex_expression(arguments[0])}}}"
    if name == "abs" and len(arguments) == 1:
        return rf"\left|{format_latex_expression(arguments[0])}\right|"
    if name == "differential" and len(arguments) == 1:
        return rf"\mathrm{{d}}{format_latex_expression(arguments[0])}"
    if name == "evaluate_at_bounds" and len(arguments) == 4:
        return _format_evaluation_at_bounds(arguments)
    if name == "evaluate_at_index" and len(arguments) == 3:
        value, variable, index = (format_latex_expression(argument) for argument in arguments)
        return rf"\left.{value}\right|_{{{variable}={index}}}"
    if name == "undefined" and not arguments:
        return r"\text{undefined}"
    if name == "integration_by_parts_rule" and not arguments:
        return r"\int u\,\mathrm{d}v = uv - \int v\,\mathrm{d}u"
    if name == "quadratic_solutions" and len(arguments) == 4:
        return _format_quadratic_solutions(arguments)
    rendered_arguments = ", ".join(format_latex_expression(item) for item in arguments)
    function = _NAMED_FUNCTIONS.get(name)
    if function is None:
        function = (
            name
            if len(name) == 1 and name.isascii() and name.isalpha()
            else rf"\operatorname{{{_escape_identifier(name)}}}"
        )
    return rf"{function}\left({rendered_arguments}\right)"


def _format_binary(expression: BinaryExpression, *, parent_precedence: int) -> str:
    operator = expression.operator
    if (
        operator is BinaryOperator.ADD
        and isinstance(expression.left, Symbol)
        and expression.left.name == "C"
    ):
        expression = BinaryExpression(
            operator=operator,
            left=expression.right,
            right=expression.left,
        )
    precedence = _PRECEDENCE[operator]
    left = format_latex_expression(expression.left, parent_precedence=precedence)
    right_precedence = precedence if operator is BinaryOperator.POWER else precedence + 1
    right = format_latex_expression(expression.right, parent_precedence=right_precedence)
    if operator is BinaryOperator.DIVIDE:
        rendered = (
            rf"\frac{{{format_latex_expression(expression.left)}}}"
            rf"{{{format_latex_expression(expression.right)}}}"
        )
    elif operator is BinaryOperator.POWER:
        rendered = rf"{left}^{{{right}}}"
    elif operator is BinaryOperator.MULTIPLY:
        rendered = rf"{left} \cdot {right}"
    else:
        if operator is BinaryOperator.SUBTRACT and right.startswith("-"):
            right = rf"\left({right}\right)"
        rendered = f"{left} {operator.value} {right}"
    if precedence < parent_precedence:
        return rf"\left({rendered}\right)"
    return rendered


def format_latex_expression(expression: Expression, *, parent_precedence: int = 0) -> str:
    """Render one StepSolver expression as LaTeX."""
    if isinstance(expression, Number):
        return _format_fraction(expression.value)
    if isinstance(expression, ApproximateNumber):
        return expression.text
    if isinstance(expression, Symbol):
        return _escape_identifier(expression.name)
    if isinstance(expression, Constant):
        constants = {"pi": r"\pi", "e": "e", "i": "i", "oo": r"\infty"}
        return constants[expression.name.value]
    if isinstance(expression, OpaqueExpression):
        escaped = expression.text.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
        return rf"\mathtt{{{escaped}}}"
    if isinstance(expression, SequenceExpression):
        if not expression.items:
            return r"\varnothing"
        items = ", ".join(format_latex_expression(item) for item in expression.items)
        return rf"\left[{items}\right]"
    if isinstance(expression, FunctionCall):
        return _format_function(expression)
    if isinstance(expression, Relation):
        operators = {
            "=": "=",
            "!=": r"\ne",
            "<": "<",
            "<=": r"\le",
            ">": ">",
            ">=": r"\ge",
        }
        return (
            f"{format_latex_expression(expression.left)} "
            f"{operators[expression.operator.value]} "
            f"{format_latex_expression(expression.right)}"
        )
    if isinstance(expression, UnaryExpression):
        operand = format_latex_expression(expression.operand, parent_precedence=40)
        if expression.operator is UnaryOperator.FACTORIAL:
            return f"{operand}!"
        return f"{expression.operator.value}{operand}"
    return _format_binary(expression, parent_precedence=parent_precedence)


def format_latex_value(value: MathValue) -> str:
    """Render a typed solver value as LaTeX."""
    if isinstance(value, NoSolutionValue):
        return r"\text{No solution}"
    if isinstance(value, ScalarValue):
        return format_latex_expression(value.expression)
    if isinstance(value, BooleanValue):
        return r"\mathrm{true}" if value.value else r"\mathrm{false}"
    if isinstance(value, SequenceValue):
        items = ", ".join(format_latex_value(item) for item in value.items)
        return rf"\left[{items}\right]"
    if isinstance(value, MappingValue):
        entries = ", ".join(
            rf"{format_latex_expression(entry.key)} \mapsto {format_latex_value(entry.value)}"
            for entry in value.entries
        )
        return rf"\left\{{{entries}\right\}}"
    rows = r" \\ ".join(
        " & ".join(format_latex_expression(item) for item in row) for row in value.rows
    )
    return rf"\begin{{bmatrix}}{rows}\end{{bmatrix}}"
