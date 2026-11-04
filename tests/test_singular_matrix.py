"""Structured singular-matrix outcomes for the inverse operation."""

import json

import pytest

from stepsolver import (
    ExactResult,
    SingularMatrixResult,
    Solver,
    format_ascii,
    solve_isolated,
    solve_payload,
    validate_solve_request,
)


@pytest.fixture
def solver() -> Solver:
    """Use the public solver configured with the default SymPy backend."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "rank_note"),
    [
        ("inverse(matrix([[1,2],[2,4]]))", "Rank: 1 < 2"),
        (
            "inverse(matrix([[1,2,3],[2,4,6],[1,1,1]]))",
            "Rank: 2 < 3",
        ),
    ],
)
def test_singular_inverse_returns_a_typed_matrix_outcome(
    solver: Solver,
    query: str,
    rank_note: str,
) -> None:
    """A zero determinant must produce a singular result, not a raw backend failure."""
    result = solver.solve(query)

    assert isinstance(result, SingularMatrixResult)
    assert "singular" in result.reason
    assert "does not exist" in result.reason
    assert tuple(step.rule for step in result.steps) == (
        "Compute the determinant",
        "Conclude the matrix is singular",
    )
    rendered = format_ascii(result)
    assert "det(matrix(" in rendered
    assert "= 0" in rendered
    assert "inverse does not exist" in rendered
    assert rank_note in rendered
    assert "Singular matrix:" in rendered
    assert "Unsolved:" not in rendered
    assert "Undefined:" not in rendered


def test_singular_inverse_payload_has_a_distinct_status(solver: Solver) -> None:
    """Serialization must expose singularity rather than collapsing to unsolved."""
    result = solver.solve("inverse(matrix([[1,2],[2,4]]))")
    assert isinstance(result, SingularMatrixResult)

    payload = solve_payload(result)
    encoded = json.loads(json.dumps(payload.as_dict()))

    assert payload.status == "singular"
    assert payload.result_latex == r"\text{No inverse: singular matrix}"
    assert encoded["status"] == "singular"
    assert encoded["reason"] == result.reason
    expected_step_count = 2
    assert len(encoded["steps"]) == expected_step_count


def test_singular_inverse_isolated_serialization_keeps_the_status() -> None:
    """The bounded agent-facing adapter must keep singularity distinct from unsolved."""
    solution = solve_isolated(validate_solve_request("inverse(matrix([[1,2],[2,4]]))"))

    assert solution.status == "singular"
    assert solution.result is None
    assert solution.latex == r"\text{No inverse: singular matrix}"
    assert solution.reason is not None
    assert "determinant is zero" in solution.reason


def test_nonsingular_inverse_still_returns_the_exact_matrix(solver: Solver) -> None:
    """The singular precheck must not intercept invertible inputs."""
    result = solver.solve("inverse(matrix([[1,2],[3,4]]))")

    assert isinstance(result, ExactResult)
    assert "Result: matrix([[-2, 1], [3/2, -1/2]])" in format_ascii(result)
