"""Exhaustiveness regressions for public result and value serialization.

Adding a new ``SolveResult`` or ``MathValue`` variant must fail these tests until every public
renderer handles it.  Without this guard a new result type can silently fall through to the
``Unsolved:`` default (``format_ascii``) or the ``unsolved`` payload status (``solve_payload``).
"""

import typing

import pytest

from stepsolver import (
    BooleanValue,
    DivergenceKind,
    DivergentResult,
    ExactResult,
    MappingEntry,
    MappingValue,
    MathValue,
    MatrixValue,
    NoSolutionValue,
    NumericResult,
    NumericRootsValue,
    ScalarValue,
    SequenceValue,
    SingularMatrixResult,
    SolutionDomain,
    SolveResult,
    UndefinedResult,
    UniversalSolutionValue,
    UnsolvedResult,
    format_ascii,
    format_value,
    parse,
    parse_expression,
    solve_payload,
)

QUERY = parse("solve(x^2 = 4, x)")
SINGULAR_RANK = 1
SINGULAR_SIZE = 2


def _union_members(alias: object) -> frozenset[type]:
    """Return the members of a PEP 695 ``type`` union alias."""
    value = getattr(alias, "__value__", None)
    assert value is not None, f"{alias!r} is not a PEP 695 union alias"
    return frozenset(typing.get_args(value))


RESULT_STATUS: dict[type[SolveResult], str] = {
    ExactResult: "exact",
    NumericResult: "numeric",
    DivergentResult: "divergent",
    UndefinedResult: "undefined",
    SingularMatrixResult: "singular",
    UnsolvedResult: "unsolved",
}

MATH_VALUES: dict[str, MathValue] = {
    "scalar": ScalarValue(expression=parse_expression("2")),
    "sequence": SequenceValue(items=(ScalarValue(expression=parse_expression("2")),)),
    "mapping": MappingValue(
        entries=(
            MappingEntry(
                key=parse_expression("x"),
                value=ScalarValue(expression=parse_expression("2")),
            ),
        )
    ),
    "matrix": MatrixValue(rows=((parse_expression("1"), parse_expression("2")),)),
    "boolean": BooleanValue(value=True),
    "no_solution": NoSolutionValue(),
    "universal": UniversalSolutionValue(
        domain=SolutionDomain.REAL,
        exclusions=(parse_expression("0"),),
    ),
    "numeric_roots": NumericRootsValue(
        roots=(parse_expression("0.5"),),
        tolerance=1e-12,
    ),
}


def _result(result_type: type[SolveResult]) -> SolveResult:
    if result_type is ExactResult:
        return ExactResult(
            query=QUERY,
            value=ScalarValue(expression=parse_expression("2")),
            steps=(),
        )
    if result_type is NumericResult:
        return NumericResult(
            query=QUERY,
            value=NumericRootsValue(roots=(parse_expression("0.5"),), tolerance=1e-12),
            steps=(),
            method="test iteration",
            residuals=(0.0,),
        )
    if result_type is DivergentResult:
        return DivergentResult(
            query=QUERY,
            kind=DivergenceKind.NONFINITE,
            reason="no finite limit",
            steps=(),
        )
    if result_type is UndefinedResult:
        return UndefinedResult(query=QUERY, reason="undefined for this input", steps=())
    if result_type is SingularMatrixResult:
        return SingularMatrixResult(
            query=QUERY,
            reason="the determinant is zero",
            steps=(),
            determinant=parse_expression("0"),
            rank=SINGULAR_RANK,
            size=SINGULAR_SIZE,
        )
    if result_type is UnsolvedResult:
        return UnsolvedResult(query=QUERY, reason="no verified answer", steps=())
    pytest.fail(f"no sample result for {result_type.__name__}")


def test_solve_result_union_matches_serialization_registry() -> None:
    """The public union and the renderer registry must list exactly the same result types."""
    assert _union_members(SolveResult) == frozenset(RESULT_STATUS)


def test_math_value_union_matches_value_registry() -> None:
    """The public value union and the sample registry must list exactly the same value types."""
    assert _union_members(MathValue) == frozenset(type(value) for value in MATH_VALUES.values())


@pytest.mark.parametrize("result_type", RESULT_STATUS)
def test_every_result_type_serializes_with_its_own_status(result_type: type[SolveResult]) -> None:
    """Every result variant must render as itself, never as the unsolved fallback."""
    result = _result(result_type)
    rendered = format_ascii(result)
    assert rendered
    if result_type is UnsolvedResult:
        assert rendered.startswith("Input: ")
    else:
        assert "Unsolved:" not in rendered
    assert solve_payload(result).status == RESULT_STATUS[result_type]


def test_singular_matrix_result_carries_determinant_rank_and_size() -> None:
    """The singular-matrix outcome must expose the facts that justify 'no inverse'."""
    result = _result(SingularMatrixResult)
    assert isinstance(result, SingularMatrixResult)
    assert result.determinant == parse_expression("0")
    assert result.rank == SINGULAR_RANK
    assert result.size == SINGULAR_SIZE
    assert "Singular matrix:" in format_ascii(result)


@pytest.mark.parametrize("name", sorted(MATH_VALUES))
def test_every_math_value_renders(name: str) -> None:
    """Every value variant must render without falling through to the matrix default."""
    rendered = format_value(MATH_VALUES[name])
    assert rendered


def test_universal_value_reports_domain_and_exclusions() -> None:
    """The universal solution must serialize its domain and exclusions explicitly."""
    rendered = format_value(MATH_VALUES["universal"])
    assert rendered == "all real numbers except 0"
