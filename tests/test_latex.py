"""LaTeX renderer regression tests."""

from fractions import Fraction

from stepsolver import ExactResult, Solver, format_latex_expression, format_latex_value
from stepsolver.ast import ApproximateNumber, Number, OpaqueExpression, SequenceExpression
from stepsolver.parser import parse_expression
from stepsolver.results import BooleanValue, NoSolutionValue, ScalarValue


def test_fraction_power_and_function_latex() -> None:
    """Arithmetic precedence and named functions should use mathematical LaTeX."""
    expression = parse_expression("sin(x)^2 + 1/2")
    assert format_latex_expression(expression) == r"\sin\left(x\right)^{2} + \frac{1}{2}"


def test_relation_and_constants_latex() -> None:
    """Relations and built-in constants should receive conventional symbols."""
    expression = parse_expression("x <= 2*pi")
    assert format_latex_expression(expression) == r"x \le 2 \cdot \pi"


def test_solution_mapping_latex() -> None:
    """Equation solutions should retain mapping structure in LaTeX."""
    result = Solver().solve("solve(x^2-4=0,x)")
    assert isinstance(result, ExactResult)
    assert (
        format_latex_value(result.value)
        == r"\left[\left\{x \mapsto -2\right\}, \left\{x \mapsto 2\right\}\right]"
    )


def test_matrix_latex() -> None:
    """Matrix results should render with a bmatrix environment."""
    result = Solver().solve("inverse(matrix([[1,2],[3,4]]))")
    assert isinstance(result, ExactResult)
    expected = r"\begin{bmatrix}-2 & 1 \\ \frac{3}{2} & \frac{-1}{2}\end{bmatrix}"
    assert format_latex_value(result.value) == expected


def test_special_expression_nodes_latex() -> None:
    """Special functions, sequences, approximations, and opaque forms should render safely."""
    assert format_latex_expression(parse_expression("sqrt(x)")) == r"\sqrt{x}"
    assert format_latex_expression(parse_expression("abs(x)")) == r"\left|x\right|"
    assert format_latex_expression(parse_expression("alpha")) == r"\alpha"
    assert format_latex_expression(parse_expression("alpha_1")) == r"alpha\_1"
    sequence = SequenceExpression(items=(Number(value=Fraction(1)), ApproximateNumber(text="1.25")))
    assert format_latex_expression(sequence) == r"\left[1, 1.25\right]"
    assert format_latex_expression(OpaqueExpression(text="O(x_{2})")) == (r"\mathtt{O(x\_\{2\})}")


def test_boolean_and_scalar_values_latex() -> None:
    """Boolean and scalar value variants should have explicit LaTeX forms."""
    assert format_latex_value(BooleanValue(value=False)) == r"\mathrm{false}"
    scalar = ScalarValue(expression=Number(value=Fraction(2, 3)))
    assert format_latex_value(scalar) == r"\frac{2}{3}"
    assert format_latex_value(NoSolutionValue()) == r"\text{No solution}"


def test_solver_operations_use_conventional_notation() -> None:
    """Operation calls should render as mathematics instead of backend function names."""
    integral = parse_expression("integrate(sin(x),x,0,pi)")
    derivative = parse_expression("diff(x^3,x,2)")
    limit = parse_expression("limit(sin(x)/x,x,0)")
    assert format_latex_expression(integral) == r"\int_{0}^{\pi} \sin\left(x\right)\,\mathrm{d}x"
    assert format_latex_expression(derivative) == (
        r"\frac{\mathrm{d}^{2}}{\mathrm{d}x^{2}}\left(x^{3}\right)"
    )
    assert format_latex_expression(limit) == (r"\lim_{x \to 0} \frac{\sin\left(x\right)}{x}")


def test_indefinite_integral_places_the_constant_last() -> None:
    """Indefinite antiderivatives should display the integration constant conventionally."""
    result = Solver().solve("integrate(x^2,x)")
    assert isinstance(result, ExactResult)
    assert format_latex_value(result.value) == r"\frac{x^{3}}{3} + C"
