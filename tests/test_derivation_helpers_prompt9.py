"""Focused branch regressions for exponential and logarithmic derivation helpers."""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from stepsolver.derivation import equations_exponential_logarithmic as eqlog

if TYPE_CHECKING:
    import pytest

_X = sp.Symbol("x")
_Y = sp.Symbol("y")
_UNSUPPORTED = "unsupported"


def test_basic_sequence_validation_rejects_non_sequences() -> None:
    """Candidate containers must be sequences of symbolic values."""
    assert not eqlog._is_basic_sequence(5)
    assert not eqlog._is_basic_sequence("x")
    assert eqlog._is_basic_sequence([sp.Integer(1)])


def test_solve_candidates_handles_backend_errors_and_empty_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed or empty solve must not be mistaken for a candidate list."""
    assert eqlog._solve_candidates(sp.Eq(sp.Integer(1), sp.Integer(2)), _X) is None

    def raise_value_error(_equation: object, _variable: object) -> object:
        raise ValueError(_UNSUPPORTED)

    monkeypatch.setattr(sp, "solve", raise_value_error)
    assert eqlog._solve_candidates(sp.Eq(_X, sp.Integer(1)), _X) is None


def test_candidate_validation_handles_simplification_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symbolic simplification failure must reject the candidate."""
    equation = sp.Eq(sp.exp(_X), sp.Integer(1))

    def raise_type_error(_value: object) -> object:
        raise TypeError(_UNSUPPORTED)

    monkeypatch.setattr(sp, "simplify", raise_type_error)
    assert not eqlog._satisfies_original(equation, _X, sp.Integer(0))


def test_log_parts_rejects_ambiguous_products() -> None:
    """A product must contain exactly one logarithm to identify its base."""
    assert eqlog._log_parts(sp.Mul(sp.log(_X), sp.log(_Y))) is None
    assert eqlog._log_parts(sp.Mul(0, sp.log(_X), evaluate=False)) is None
    assert eqlog._log_parts(sp.Mul(sp.I, sp.log(_X), evaluate=False)) is None
    assert eqlog._log_parts(sp.log(_X)) == (sp.E, _X)


def test_split_log_sum_covers_product_quotient_and_invalid_forms() -> None:
    """The splitter must distinguish sums, differences, and non-log terms."""
    assert eqlog._split_log_sum(sp.log(_X) + sp.Integer(1)) is None
    assert eqlog._split_log_sum(sp.log(_X) + sp.log(_X, sp.Integer(2))) is None
    assert eqlog._split_log_sum(-sp.log(_X) - sp.log(_Y)) is None
    quotient = sp.Add(sp.log(_X), sp.Mul(-1, sp.log(_Y), evaluate=False), evaluate=False)
    assert eqlog._split_log_sum(quotient) == (sp.E, sp.Mul(_X, sp.Pow(_Y, -1)))


def test_single_log_side_rejects_nested_and_log_free_equations() -> None:
    """Only one side may contain a single logarithm."""
    assert eqlog._single_log_side(sp.Eq(sp.log(_X), sp.log(_Y))) is None
    assert eqlog._single_log_side(sp.Eq(sp.Symbol("a"), sp.Symbol("b"), evaluate=False)) is None


def test_derive_from_log_guards_against_failed_or_mismatched_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derivation must abort if solving fails or root sets disagree."""
    equation = sp.Eq(sp.log(_X), sp.Integer(0), evaluate=False)
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: None)
    assert eqlog._derive_from_log(equation, _X, (sp.Integer(1),), sp.E, _X, sp.Integer(0)) == ()
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: (sp.Integer(1),))
    assert eqlog._derive_from_log(equation, _X, (), sp.E, _X, sp.Integer(0)) == ()


def test_log_strategies_abort_for_unrelated_equations() -> None:
    """Each strategy must decline equations outside its structural family."""
    empty_equation = sp.Eq(sp.Symbol("a"), sp.Symbol("b"), evaluate=False)
    assert eqlog._derive_single_log(empty_equation, _X, ()) == ()
    assert (
        eqlog._derive_log_sum(
            sp.Eq(sp.log(_X) + sp.log(_X + 1), _X, evaluate=False),
            _X,
            (),
        )
        == ()
    )
    assert eqlog._derive_equal_logs(sp.Eq(sp.log(_X), sp.log(_X, sp.Integer(2))), _X, ()) == ()
    assert eqlog._derive_log_equation(sp.Eq(_X + 1, sp.Integer(0)), _X, ()) == ()


def test_constant_and_log_requires_one_logarithm_and_one_constant() -> None:
    """Three-term sums and log-only sums cannot be parsed as log plus constant."""
    assert eqlog._constant_and_log(sp.log(_X) + sp.log(_Y) + sp.Integer(1), _X) is None
    assert eqlog._constant_and_log(sp.log(_X) + sp.log(_Y), _X) is None
    assert eqlog._constant_and_log(sp.log(_X) + _X, _X) is None
    assert eqlog._constant_and_log(sp.log(_X) + sp.Integer(1), _X) == (
        (sp.E, _X),
        sp.Integer(1),
    )


def test_log_with_constant_parts_requires_matching_bases() -> None:
    """A constant offset is usable only when both logarithms share a base."""
    assert (
        eqlog._log_with_constant_parts(
            sp.Eq(sp.log(_X, sp.Integer(2)), sp.log(_X) + sp.Integer(1), evaluate=False),
            _X,
        )
        is None
    )


def test_derive_single_exponential_guards_against_failed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single exponential must abort when solving fails or roots disagree."""
    equation = sp.Eq(sp.exp(_X), sp.Integer(2), evaluate=False)
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: None)
    assert eqlog._derive_single_exponential(equation, _X, ()) == ()
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: (sp.log(sp.Integer(2)),))
    assert eqlog._derive_single_exponential(equation, _X, ()) == ()


def test_derive_equal_exponentials_guards_against_failed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal exponentials must abort when solving fails or roots disagree."""
    equation = sp.Eq(sp.exp(_X), sp.exp(2 * _X), evaluate=False)
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: None)
    assert eqlog._derive_equal_exponentials(equation, _X, ()) == ()
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: (sp.Integer(0),))
    assert eqlog._derive_equal_exponentials(equation, _X, ()) == ()


def test_positive_slopes_rejects_nonconstant_and_incommensurate_exponents() -> None:
    """Only constant integer-compatible exponential rates may be substituted."""
    assert eqlog._positive_slopes((sp.exp(_X**2), sp.exp(_X)), _X) is None
    assert eqlog._positive_slopes((sp.exp(sp.pi * _X), sp.exp(_X)), _X) is None
    assert eqlog._positive_slopes((sp.exp(2 * _X), sp.exp(_X)), _X) == (
        sp.Integer(1),
        ((sp.exp(2 * _X), sp.Integer(2)), (sp.exp(_X), sp.Integer(1))),
    )


def test_prime_exponents_reports_denominators_as_negative_powers() -> None:
    """A rational base contributes negative exponents for its denominator primes."""
    assert eqlog._prime_exponents(sp.Rational(1, 2)) == {sp.Integer(2): -1}
    assert eqlog._prime_exponents(sp.Rational(12, 5)) == {
        sp.Integer(2): 2,
        sp.Integer(3): 1,
        sp.Integer(5): -1,
    }


def test_common_power_base_rejects_incompatible_factorizations() -> None:
    """A common base exists only when every base shares normalized exponents."""
    assert eqlog._common_power_base(()) is None
    assert eqlog._common_power_base(({},)) is None
    assert eqlog._common_power_base(({sp.Integer(2): 0},)) is None
    assert (
        eqlog._common_power_base(
            (
                {sp.Integer(2): 1},
                {sp.Integer(3): 1},
            )
        )
        is None
    )
    assert eqlog._common_power_base(
        (
            {sp.Integer(2): 2},
            {sp.Integer(2): 4},
        )
    ) == (sp.Integer(2), (2, 4))


def test_detect_power_substitution_rejects_nonconstant_and_mixed_bases() -> None:
    """Variable-base substitution requires constant, compatible exponents."""
    assert eqlog._detect_power_substitution(sp.Eq(_X**2 + 2**_X, 0), _X) == (
        sp.Integer(2),
        sp.Integer(1),
        ((sp.Pow(sp.Integer(2), _X, evaluate=False), sp.Integer(1)),),
    )
    assert eqlog._detect_power_substitution(sp.Eq(2 ** (_X**2) + 2**_X, 0), _X) == (
        sp.Integer(2),
        sp.Integer(1),
        ((sp.Pow(sp.Integer(2), _X, evaluate=False), sp.Integer(1)),),
    )
    assert eqlog._detect_power_substitution(sp.Eq(_X + 1, 0), _X) is None
    assert eqlog._detect_power_substitution(sp.Eq(2**_X + 3**_X, 0), _X) is None
    assert (
        eqlog._detect_power_substitution(
            sp.Eq(2 ** (sp.sqrt(sp.Integer(2)) * _X) + 4 ** (sp.sqrt(sp.Integer(2)) * _X), 0),
            _X,
        )
        is None
    )


def test_exponential_substitution_rejects_unsupported_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The substitution strategy must decline shapes it cannot reduce quadratically."""
    assert (
        eqlog._derive_exponential_substitution(sp.Eq(sp.exp(_X**2) + sp.exp(_X), 0), _X, ()) == ()
    )
    assert eqlog._derive_exponential_substitution(sp.Eq(_X, 0), _X, ()) == ()
    assert eqlog._derive_exponential_substitution(sp.Eq(2**_X + _X, 0), _X, ()) == ()
    assert eqlog._derive_exponential_substitution(sp.Eq(2**_X - 1, 0), _X, ()) == ()
    quadratic = sp.Eq(2 ** (2 * _X) - 5 * 2**_X + 4, 0)
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: None)
    assert eqlog._derive_exponential_substitution(quadratic, _X, ()) == ()
    monkeypatch.setattr(eqlog, "_solve_candidates", lambda *_: (sp.Integer(1), sp.Integer(4)))
    assert eqlog._derive_exponential_substitution(quadratic, _X, ()) == ()
    assert eqlog._derive_exponential_substitution(sp.Eq(2 ** (2 * _X) + 1, 0), _X, ()) == ()
    monkeypatch.setattr(eqlog, "_satisfies_original", lambda *_: False)
    assert (
        eqlog._derive_exponential_substitution(quadratic, _X, (sp.Integer(0), sp.Integer(2))) == ()
    )
