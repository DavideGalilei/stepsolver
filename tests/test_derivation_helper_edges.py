"""Edge-branch coverage for the Prompt 7 integration and inequality helpers."""

# pyright: reportPrivateUsage=false

import sympy as sp

from stepsolver.derivation.definite_integrals_improper import (
    _derive_absolute_value_improper_integral,
    _derive_endpoint_singularity_integral,
    _derive_two_sided_infinite_integral,
)
from stepsolver.derivation.integrals_elementary import _derive_scaled_inverse_sine
from stepsolver.derivation.integrals_special import derive_polynomial_inverse_by_parts
from stepsolver.parser import parse, parse_expression
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_equation_steps import SympyEquationStepBuilder
from stepsolver.sympy_rendering import SympyDerivationRenderer
from stepsolver.sympy_steps import SympyStepBuilder, _derivative_order


def _equation_builder() -> SympyEquationStepBuilder:
    return SympyEquationStepBuilder(SympyConverter())


def _step_builder() -> SympyStepBuilder:
    converter = SympyConverter()
    return SympyStepBuilder(converter, SympyDerivationRenderer(converter))


def test_scaled_inverse_sine_declines_unrecognized_shapes() -> None:
    """Only a constant multiple of 1/sqrt(a^2 - (b*x)^2) may use the pattern."""
    variable = sp.Symbol("x", real=True)
    wrong_result = sp.Integer(0)
    assert _derive_scaled_inverse_sine(sp.sin(variable), variable, wrong_result) == ()
    assert _derive_scaled_inverse_sine(1 / (4 - variable**2), variable, wrong_result) == ()
    assert _derive_scaled_inverse_sine(1 / sp.sqrt(4 + variable**2), variable, wrong_result) == ()
    assert _derive_scaled_inverse_sine(1 / sp.sqrt(-4 - variable**2), variable, wrong_result) == ()


def test_scaled_inverse_sine_rejects_a_mismatched_result() -> None:
    """A matching shape still needs the displayed antiderivative to be equivalent."""
    variable = sp.Symbol("x", real=True)
    assert _derive_scaled_inverse_sine(1 / sp.sqrt(4 - variable**2), variable, sp.Integer(0)) == ()


def test_polynomial_inverse_by_parts_declines_ineligible_integrands() -> None:
    """The by-parts rule needs one inverse factor and a genuine polynomial partner."""
    variable = sp.Symbol("x", real=True)
    wrong_result = sp.Integer(0)
    assert derive_polynomial_inverse_by_parts(sp.sin(variable), variable, wrong_result) == ()
    assert (
        derive_polynomial_inverse_by_parts(sp.log(variable) / variable, variable, wrong_result)
        == ()
    )
    assert derive_polynomial_inverse_by_parts(2 * sp.log(variable), variable, wrong_result) == ()
    assert (
        derive_polynomial_inverse_by_parts(variable * sp.log(variable), variable, wrong_result)
        == ()
    )


def test_polynomial_inverse_by_parts_handles_a_negative_remaining_term() -> None:
    """A negative remaining integrand should be shown as an added positive integral."""
    variable = sp.Symbol("x", real=True)
    integrand = -variable * sp.log(variable)
    result = sp.integrate(integrand, variable) + sp.Symbol("C")
    derivation = derive_polynomial_inverse_by_parts(integrand, variable, result)
    assert len(derivation) == 2
    assert derivation[0].rule == "Choose integration by parts"


def test_inequality_helpers_reject_unresolved_comparisons() -> None:
    """Symbolic or unknown comparisons must not be reported as true or false."""
    builder = _equation_builder()
    symbolic = sp.Symbol("a")
    assert builder._comparison_holds("==", sp.Integer(1)) is None
    assert builder._comparison_holds(">", symbolic) is None
    assert builder._test_intervals("==", sp.Symbol("x"), sp.Symbol("x"), [sp.Integer(0)]) == []


def test_real_solutions_handles_constants_and_unknown_reality() -> None:
    """A constant has no zeros, while an unknown reality cannot be charted."""
    builder = _equation_builder()
    variable = sp.Symbol("x", real=True)
    assert builder._real_solutions(sp.Integer(1), variable) == ()
    assert builder._real_solutions(variable**2 + sp.Symbol("a"), variable) is None


def test_boundary_explanations_cover_polynomial_and_absolute_value_forms() -> None:
    """Each boundary type should explain why only its zeros can change the sign."""
    builder = _equation_builder()
    variable = sp.Symbol("x", real=True)
    assert "A polynomial can change sign" in builder._boundary_explanation(
        sp.Integer(1), has_poles=False
    )
    assert "An absolute-value expression can change sign" in builder._boundary_explanation(
        sp.Abs(variable) - 1,
        has_poles=False,
    )


def test_sign_chart_reports_numerator_zeros_and_denominator_poles() -> None:
    """The sign chart separates numerator zeros from denominator poles."""
    builder = _equation_builder()
    variable = sp.Symbol("x", real=True)
    assert builder._sign_chart(sp.exp(variable), variable) == ((), ())
    chart = builder._sign_chart((variable - 1) / (variable + 2), variable)
    assert chart is not None
    zeros, poles = chart
    assert zeros == (sp.Integer(1),)
    assert poles == (sp.Integer(-2),)


def test_two_sided_infinite_integral_declines_ineligible_inputs() -> None:
    """Symmetry is only used for a whole-line integral with a matching finite value."""
    variable = sp.Symbol("x", real=True)
    assert (
        _derive_two_sided_infinite_integral(
            variable / (1 + variable**2),
            variable,
            -sp.oo,
            sp.oo,
            sp.Integer(0),
        )
        == ()
    )
    assert (
        _derive_two_sided_infinite_integral(
            1 / (1 + variable**2),
            variable,
            -sp.oo,
            sp.oo,
            sp.Integer(0),
        )
        == ()
    )
    assert (
        _derive_two_sided_infinite_integral(variable, variable, -sp.oo, sp.oo, sp.Integer(0)) == ()
    )
    assert (
        _derive_two_sided_infinite_integral(
            1 / (1 + variable**2),
            variable,
            sp.Integer(0),
            sp.oo,
            sp.pi / 2,
        )
        == ()
    )


def test_endpoint_singularity_declines_unsupported_intervals() -> None:
    """Only a single finite endpoint singularity with a verified limit is described."""
    variable = sp.Symbol("x", real=True)
    assert (
        _derive_endpoint_singularity_integral(
            1 / (variable * (variable - 2)),
            variable,
            sp.Integer(0),
            sp.Integer(3),
            sp.Integer(0),
        )
        == ()
    )
    assert (
        _derive_endpoint_singularity_integral(
            1 / (variable - 1),
            variable,
            sp.Integer(0),
            sp.Integer(2),
            sp.Integer(0),
        )
        == ()
    )
    assert (
        _derive_endpoint_singularity_integral(
            1 / variable, variable, sp.I, sp.Integer(0), sp.Integer(0)
        )
        == ()
    )
    assert (
        _derive_endpoint_singularity_integral(
            1 / sp.sqrt(variable),
            variable,
            sp.Integer(0),
            sp.Integer(1),
            sp.Integer(0),
        )
        == ()
    )
    assert (
        _derive_endpoint_singularity_integral(
            1 / variable, variable, sp.Integer(0), sp.oo, sp.Integer(0)
        )
        == ()
    )


def test_absolute_value_improper_integral_declines_nonstandard_bounds() -> None:
    """The special absolute-value derivation is limited to the two standard ranges."""
    variable = sp.Symbol("x", real=True)
    assert (
        _derive_absolute_value_improper_integral(
            sp.Abs(variable), variable, sp.Integer(0), sp.Integer(1)
        )
        == ()
    )


def test_derivative_order_rejects_non_numeric_and_invalid_orders() -> None:
    """A derivative order must be a positive whole number."""
    assert _derivative_order(parse("diff(x^4,x,y)")) is None
    assert _derivative_order(parse("diff(x^4,x,1/2)")) is None
    assert _derivative_order(parse("diff(x^4,x,0)")) is None
    assert _derivative_order(parse("diff(x^4,x,2)")) == 2


def test_integration_domain_constraints_requires_a_scalar_integrand() -> None:
    """A non-scalar input cannot contribute denominator exclusions."""
    builder = _step_builder()
    assert builder.integration_domain_constraints(parse_expression("matrix([[1,2]])")) == ()


def test_solution_roots_rejects_non_mapping_backend_values() -> None:
    """Only a sequence of symbol mappings can supply target roots."""
    variable = sp.Symbol("x", real=True)
    builder = _step_builder()
    assert builder.solution_roots(sp.Integer(1), variable) == ()
    assert builder.solution_roots([sp.Integer(1)], variable) == ()
    assert builder.solution_roots([{sp.Symbol("y"): sp.Integer(1)}], variable) == ()
