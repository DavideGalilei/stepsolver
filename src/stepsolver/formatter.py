"""ASCII formatting for syntax trees and solver results."""

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
    DivergentResult,
    ExactResult,
    MappingValue,
    MathValue,
    ScalarValue,
    SequenceValue,
    SolveResult,
)

_PRECEDENCE: dict[BinaryOperator, int] = {
    BinaryOperator.ADD: 10,
    BinaryOperator.SUBTRACT: 10,
    BinaryOperator.MULTIPLY: 20,
    BinaryOperator.DIVIDE: 20,
    BinaryOperator.POWER: 30,
}


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_expression(expression: Expression, *, parent_precedence: int = 0) -> str:
    """Render a mathematical expression using parseable ASCII syntax."""
    if isinstance(expression, Number):
        return _format_fraction(expression.value)
    if isinstance(expression, ApproximateNumber):
        return expression.text
    if isinstance(expression, Symbol):
        return expression.name
    if isinstance(expression, Constant):
        return expression.name.value
    if isinstance(expression, OpaqueExpression):
        return expression.text
    if isinstance(expression, SequenceExpression):
        return f"[{', '.join(format_expression(item) for item in expression.items)}]"
    if isinstance(expression, FunctionCall):
        arguments = ", ".join(format_expression(item) for item in expression.arguments)
        return f"{expression.name}({arguments})"
    if isinstance(expression, Relation):
        return (
            f"{format_expression(expression.left)} {expression.operator.value} "
            f"{format_expression(expression.right)}"
        )
    if isinstance(expression, UnaryExpression):
        operand = format_expression(expression.operand, parent_precedence=40)
        if expression.operator is UnaryOperator.FACTORIAL:
            return f"{operand}!"
        return f"{expression.operator.value}{operand}"
    if (
        expression.operator is BinaryOperator.ADD
        and isinstance(expression.left, Symbol)
        and expression.left.name == "C"
    ):
        expression = BinaryExpression(
            operator=BinaryOperator.ADD,
            left=expression.right,
            right=expression.left,
        )
    precedence = _PRECEDENCE[expression.operator]
    left = format_expression(expression.left, parent_precedence=precedence)
    right_precedence = precedence if expression.operator is BinaryOperator.POWER else precedence + 1
    right = format_expression(expression.right, parent_precedence=right_precedence)
    rendered = f"{left} {expression.operator.value} {right}"
    if precedence < parent_precedence:
        return f"({rendered})"
    return rendered


def _format_value(value: MathValue) -> str:
    if isinstance(value, ScalarValue):
        return format_expression(value.expression)
    if isinstance(value, BooleanValue):
        return "true" if value.value else "false"
    if isinstance(value, SequenceValue):
        return f"[{', '.join(_format_value(item) for item in value.items)}]"
    if isinstance(value, MappingValue):
        entries = ", ".join(
            f"{format_expression(entry.key)}: {_format_value(entry.value)}"
            for entry in value.entries
        )
        return f"{{{entries}}}"
    rows = ", ".join(
        f"[{', '.join(format_expression(item) for item in row)}]" for row in value.rows
    )
    return f"matrix([{rows}])"


def format_ascii(result: SolveResult) -> str:
    """Render a complete solve result as readable ASCII text."""
    lines = [f"Input: {result.query.source}"]
    for index, step in enumerate(result.steps, start=1):
        lines.extend(
            (
                f"Step {index} ({step.rule}): {step.explanation}",
                f"  {format_expression(step.before)}",
                f"  -> {format_expression(step.after)}",
                f"  Verified by {step.verification.method.value}: {step.verification.detail}",
            )
        )
        lines.extend(f"  {note.label}: {format_expression(note.expression)}" for note in step.notes)
    if isinstance(result, ExactResult):
        lines.append(f"Result: {_format_value(result.value)}")
    elif isinstance(result, DivergentResult):
        lines.append(f"Diverges: {result.reason}")
    else:
        lines.append(f"Unsolved: {result.reason}")
    return "\n".join(lines)
