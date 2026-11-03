"""The native ``ln`` spelling maps to the existing natural logarithm."""

import pytest

from stepsolver import ExactResult, ParseError, QueryError, Solver, format_ascii, parse_expression


def test_ln_is_the_same_function_as_log() -> None:
    """``ln(x)`` should produce the existing ``log(x)`` AST."""
    assert parse_expression("ln(x)") == parse_expression("log(x)")


def test_ln_rejects_a_second_argument() -> None:
    """``ln`` is natural log only; it must not become a base-2 logarithm."""
    with pytest.raises(ParseError, match="ln expects one argument"):
        parse_expression("ln(x,2)")


def test_e05_simplifies_native_ln_without_adapter_rewriting() -> None:
    """The benchmark E05 input should work directly."""
    result = Solver().solve("simplify(ln(e^x))")
    assert isinstance(result, ExactResult)
    assert format_ascii(result).rstrip().endswith("Result: x")


def test_f05_differentiates_native_ln_with_a_domain_constraint() -> None:
    """The benchmark F05 input should work directly and state the domain."""
    result = Solver().solve("diff(ln(x),x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert rendered.rstrip().endswith("Result: 1 / x")
    assert "Domain constraint: x > 0" in rendered


def test_native_ln_works_inside_a_compound_argument() -> None:
    """The alias should compose with the ordinary function pipeline."""
    result = Solver().solve("diff(ln(x^2+1),x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert rendered.rstrip().endswith("Result: 2 * x / (x ^ 2 + 1)")
    assert "Domain constraint: x ^ 2 + 1 > 0" in rendered


@pytest.mark.parametrize("query", ["lnsin(x)", "sinln(x)", "frobnicate(x)"])
def test_ln_prefixes_and_unknown_functions_stay_unsupported(query: str) -> None:
    """Only the exact identifier ``ln`` should be recognized."""
    with pytest.raises(QueryError, match="Unsupported function"):
        Solver().solve(query)
