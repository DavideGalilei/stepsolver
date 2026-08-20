"""Focused backend derivation strategy tests."""

import pytest
import sympy as sp

from stepsolver.derivation.integrals_structural import (
    derive_cyclic_exponential_trig_integral,
    derive_inverse_function_by_parts,
    derive_inverse_tangent_substitution,
    derive_trig_power_substitution,
)
from stepsolver.derivation.sums import derive_sum
from stepsolver.sympy_derivation import (
    derive_basic_antiderivative,
    derive_dirichlet_integral,
    derive_reciprocal_quadratic_integral,
)


def test_nth_term_strategy_declines_when_the_test_cannot_prove_divergence() -> None:
    """Finite bounds, zero term limits, and unresolved limits need another series method."""
    variable = sp.Symbol("n", integer=True, positive=True)
    expression = sp.sin(variable**2) / (variable + 1)
    assert derive_sum(expression, variable, sp.Integer(1), sp.Integer(10), sp.Integer(0)) == ()
    assert derive_sum(expression, variable, sp.Integer(1), sp.oo, sp.Integer(0)) == ()
    unknown_term = sp.Function("f")(variable)
    unevaluated_sum = sp.summation(unknown_term, (variable, sp.Integer(1), sp.oo))
    assert derive_sum(unknown_term, variable, sp.Integer(1), sp.oo, unevaluated_sum) == ()


def test_structural_integral_strategies_reject_unrelated_integrands() -> None:
    """Stateless method recognizers should fail closed when their pattern is absent."""
    variable = sp.Symbol("x", real=True)
    unrelated = sp.sin(variable) + variable
    result = sp.integrate(unrelated, variable) + sp.Symbol("C")
    assert derive_inverse_function_by_parts(unrelated, variable, result) == ()
    assert derive_trig_power_substitution(unrelated, variable, result) == ()
    assert derive_inverse_tangent_substitution(unrelated, variable, result) == ()
    assert derive_cyclic_exponential_trig_integral(unrelated, variable, result) == ()


def test_structural_integral_strategies_reject_unverified_near_matches() -> None:
    """Recognized shapes still need constant rates and the exact expected result."""
    variable = sp.Symbol("x", real=True)
    integration_constant = sp.Symbol("C")

    assert (
        derive_inverse_function_by_parts(
            sp.log(variable),
            variable,
            variable + integration_constant,
        )
        == ()
    )
    assert (
        derive_trig_power_substitution(
            sp.sin(variable) ** 2 * sp.cos(variable),
            variable,
            variable + integration_constant,
        )
        == ()
    )
    assert (
        derive_inverse_tangent_substitution(
            sp.Rational(1, 2),
            variable,
            variable / 2 + integration_constant,
        )
        == ()
    )
    assert (
        derive_inverse_tangent_substitution(
            1 / (variable**2 + 1),
            variable,
            sp.atan(variable) + integration_constant,
        )
        == ()
    )
    assert (
        derive_cyclic_exponential_trig_integral(
            sp.exp(variable**2) * sp.sin(variable),
            variable,
            sp.integrate(sp.exp(variable**2) * sp.sin(variable), variable),
        )
        == ()
    )
    assert (
        derive_cyclic_exponential_trig_integral(
            sp.exp(variable) * sp.sin(variable),
            variable,
            variable + integration_constant,
        )
        == ()
    )


def test_integral_strategy_declines_unsupported_rational_forms() -> None:
    """Only constant-over-positive-quadratic integrands should use this strategy."""
    variable = sp.Symbol("x", real=True)
    assert (
        derive_reciprocal_quadratic_integral(
            variable / (variable**2 + 1),
            variable,
            sp.log(variable**2 + 1) / 2,
        )
        == ()
    )
    assert (
        derive_reciprocal_quadratic_integral(
            1 / (variable + 1),
            variable,
            sp.log(variable + 1),
        )
        == ()
    )
    assert (
        derive_reciprocal_quadratic_integral(
            1 / (variable**2 - 1),
            variable,
            sp.log((variable - 1) / (variable + 1)) / 2,
        )
        == ()
    )


def test_integral_strategy_rejects_an_incorrect_expected_result() -> None:
    """A derivation must agree with the exact backend antiderivative."""
    variable = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="differs from the exact result"):
        derive_reciprocal_quadratic_integral(
            1 / (variable**2 + 1),
            variable,
            variable,
        )


def test_dirichlet_strategy_only_accepts_the_verified_improper_integral() -> None:
    """The damping derivation should not be applied to superficially similar integrals."""
    variable = sp.Symbol("x", real=True)
    steps = derive_dirichlet_integral(
        sp.sin(variable) / variable,
        variable,
        sp.Integer(0),
        sp.oo,
        sp.pi / 2,
    )
    assert tuple(step.rule for step in steps) == (
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    )
    assert (
        derive_dirichlet_integral(
            sp.cos(variable) / variable,
            variable,
            sp.Integer(0),
            sp.oo,
            sp.pi / 2,
        )
        == ()
    )


def test_basic_integral_strategies_reject_nonmatching_forms_and_results() -> None:
    """Elementary rules should decline nonlinear chains, sums, and incorrect results."""
    variable = sp.Symbol("x", real=True)
    assert derive_basic_antiderivative(sp.sin(variable), variable, variable) == ()
    assert derive_basic_antiderivative(sp.sin(variable**2), variable, variable) == ()
    assert derive_basic_antiderivative(variable**2 + variable, variable, variable) == ()
    assert derive_basic_antiderivative(variable**2, variable, variable) == ()


def test_dirichlet_strategy_rejects_invalid_frequency_bounds_and_result() -> None:
    """The improper-integral proof should require its sign, interval, and exact value."""
    variable = sp.Symbol("x", real=True)
    assert (
        derive_dirichlet_integral(
            sp.sin(-2 * variable) / variable,
            variable,
            sp.Integer(0),
            sp.oo,
            -sp.pi / 2,
        )
        == ()
    )
    assert (
        derive_dirichlet_integral(
            sp.sin(variable) / variable,
            variable,
            sp.Integer(1),
            sp.oo,
            sp.pi / 2,
        )
        == ()
    )
    assert (
        derive_dirichlet_integral(
            sp.sin(variable) / variable,
            variable,
            sp.Integer(0),
            sp.oo,
            sp.Integer(0),
        )
        == ()
    )
