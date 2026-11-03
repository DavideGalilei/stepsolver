"""Edge-case coverage for the advanced derivation builders.

The public solver rejects malformed inputs before these builders run.  These tests exercise
the builders' own defensive checks so a computed backend object cannot be presented as a
verified derivation when the input or backend value is outside the documented scope.
"""

import sympy as sp

from stepsolver.ast import Operation, Query
from stepsolver.parser import parse, parse_expression
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_equation_steps import SympyEquationStepBuilder
from stepsolver.sympy_matrix_steps import SympyMatrixStepBuilder


def test_matrix_builder_rejects_non_matrix_and_arity_mismatches() -> None:
    """The matrix builder should decline queries outside its dispatch table."""
    builder = SympyMatrixStepBuilder(SympyConverter())
    assert builder.steps(parse("integrate(x,x)"), sp.Integer(1)) == ()
    bad_query = Query(operation=Operation.MATRIX, arguments=(), source="matrix()")
    assert builder.steps(bad_query, sp.eye(1)) == ()


def test_matrix_builder_rejects_unverified_backend_values() -> None:
    """Every matrix operation must compare its recomputed value with the backend."""
    builder = SympyMatrixStepBuilder(SympyConverter())
    assert (
        builder.steps(parse("matrix([[1,2]])"), sp.Matrix([[sp.Integer(1)], [sp.Integer(2)]])) == ()
    )
    assert (
        builder.steps(parse("matrix([[1,2]])"), sp.Matrix([[sp.Integer(1), sp.Integer(3)]])) == ()
    )
    assert builder.steps(parse("det(matrix([[1,2,3],[4,5,6]]))"), sp.Integer(0)) == ()
    assert builder.steps(parse("det(matrix([[5]]))"), sp.Integer(6)) == ()
    assert builder.steps(parse("det(matrix([[1,2],[3,4]]))"), sp.Integer(-3)) == ()
    assert (
        builder.steps(
            parse("det(matrix([[1,2,3],[0,1,4],[5,6,0]]))"),
            sp.Integer(2),
        )
        == ()
    )
    assert (
        builder.steps(
            parse("inverse(matrix([[1,2,3],[4,5,6]]))"),
            sp.eye(2),
        )
        == ()
    )
    assert builder.steps(parse("inverse(matrix([[1,2],[2,4]]))"), sp.eye(2)) == ()
    assert builder.steps(parse("rank(matrix([[1,2],[2,4]]))"), 2) == ()
    assert (
        builder.steps(
            parse("eigenvalues(matrix([[1,2,3],[4,5,6]]))"),
            {},
        )
        == ()
    )
    assert (
        builder.steps(
            parse("eigenvalues(matrix([[2,0],[0,3]]))"),
            {sp.Integer(2): 1.0},
        )
        == ()
    )
    assert (
        builder.steps(
            parse("eigenvalues(matrix([[2,0],[0,3]]))"),
            {sp.Integer(2): 1, sp.Integer(4): 1},
        )
        == ()
    )


def test_matrix_builder_handles_a_single_row_with_extra_columns() -> None:
    """A matrix with more columns than rows must stop after its last pivot row."""
    builder = SympyMatrixStepBuilder(SympyConverter())
    query = parse("rref(matrix([[1,2]]))")
    backend_value = sp.Matrix([[sp.Integer(1), sp.Integer(2)]]).rref()
    steps = builder.steps(query, backend_value)
    assert steps
    assert steps[-1].rule == "Read the reduced row echelon form"


def test_equation_builder_rejects_wrong_operations_and_backend_shapes() -> None:
    """The equation builder should guard its arity, dispatch, and backend type."""
    builder = SympyEquationStepBuilder(SympyConverter())
    bad_arity = Query(
        operation=Operation.DSOLVE,
        arguments=(parse_expression("y(x)"),),
        source="dsolve(y(x))",
    )
    assert builder.steps(bad_arity, sp.Integer(0)) == ()
    assert builder.steps(parse("integrate(x,x)"), sp.Integer(1)) == ()
    assert builder.steps(parse("dsolve(diff(y(x),x)=y(x),y(x))"), sp.Integer(0)) == ()


def test_equation_builder_rejects_unverified_ode_families() -> None:
    """An ODE family without the required constants must not be accepted."""
    converter = SympyConverter()
    builder = SympyEquationStepBuilder(converter)
    query = parse("dsolve(diff(y(x),x)=y(x),y(x))")
    function = converter.to_sympy(parse_expression("y(x)"))
    variable = converter.to_sympy(parse_expression("x"))
    assert builder.steps(query, sp.Eq(function, sp.Integer(0))) == ()
    assert builder.steps(query, sp.Eq(function, sp.exp(variable))) == ()


def test_equation_builder_rejects_malformed_recurrences() -> None:
    """A recurrence builder must require a sequence application and matching terms."""
    converter = SympyConverter()
    builder = SympyEquationStepBuilder(converter)
    recurrence = parse_expression("a(n+1)-a(n)=1")
    wrong_term = Query(
        operation=Operation.RSOLVE,
        arguments=(recurrence, parse_expression("x")),
        source="rsolve(a(n+1)-a(n)=1,x)",
    )
    assert builder.steps(wrong_term, sp.Integer(0)) == ()
    different_function = Query(
        operation=Operation.RSOLVE,
        arguments=(recurrence, parse_expression("y(n)")),
        source="rsolve(a(n+1)-a(n)=1,y(n))",
    )
    assert builder.steps(different_function, sp.Integer(0)) == ()


def test_equation_builder_handles_constant_sign_and_rejects_mismatches() -> None:
    """A constant-sign inequality is explained, while mismatched sets stay unexplained."""
    builder = SympyEquationStepBuilder(SympyConverter())
    assert (
        builder.steps(
            parse("solve_inequality(sin(x)<0,x)"),
            sp.Interval.open(sp.Integer(-1), sp.Integer(1)),
        )
        == ()
    )
    constant_steps = builder.steps(parse("solve_inequality(x^2+1>0,x)"), sp.true)
    assert tuple(step.rule for step in constant_steps) == (
        "Move every term to one side",
        "Check the sign for every real number",
    )
    wrong_set = sp.Lt(sp.Symbol("x", real=True), sp.Integer(1))
    assert builder.steps(parse("solve_inequality(x^2<4,x)"), wrong_set) == ()
