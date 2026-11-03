"""Regression coverage for advanced operations that used to collapse to one opaque step."""

import pytest

from stepsolver import (
    ExactResult,
    Solver,
    VerificationMethod,
    format_ascii,
    format_latex_expression,
    solve_payload,
)

_OPAQUE_RULE = "Compute exact result"
_THREE_BY_THREE_MINORS = 3
_TWO_ELIMINATIONS = 2


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


# These nine documented operation families are the advanced cases that currently
# return a single "Compute exact result" placeholder.  The upstream benchmark IDs
# (A05/A07/C07/K02) are not present in this revision, so the table records the
# repository inputs and their mathematical expectations directly.
_ADVANCED_CASES = (
    (
        "ODE-1",
        "dsolve(diff(y(x),x)=y(x),y(x))",
        "Result: Eq(y(x), C1 * exp(x))",
    ),
    (
        "RECURRENCE-1",
        "rsolve(a(n+1)-a(n)=1,a(n))",
        "Result: C0 + n",
    ),
    (
        "MATRIX-1",
        "matrix([[1,2],[3,4]])",
        "Result: matrix([[1, 2], [3, 4]])",
    ),
    (
        "DETERMINANT-1",
        "det(matrix([[1,2],[3,4]]))",
        "Result: -2",
    ),
    (
        "INVERSE-1",
        "inverse(matrix([[1,2],[3,4]]))",
        "Result: matrix([[-2, 1], [3/2, -1/2]])",
    ),
    (
        "RANK-1",
        "rank(matrix([[1,2],[2,4]]))",
        "Result: 1",
    ),
    (
        "RREF-1",
        "rref(matrix([[1,2],[2,4]]))",
        "Result: [matrix([[1, 2], [0, 0]]), [0]]",
    ),
    (
        "EIGENVALUES-1",
        "eigenvalues(matrix([[2,0],[0,3]]))",
        "Result: {2: 1, 3: 1}",
    ),
    (
        "INEQUALITY-1",
        "solve_inequality(x^2<4,x)",
        "Result: (-2 < x) & (x < 2)",
    ),
)


@pytest.mark.parametrize(("case_id", "query", "expected_result"), _ADVANCED_CASES)
def test_advanced_operation_has_student_facing_steps(
    solver: Solver,
    case_id: str,
    query: str,
    expected_result: str,
) -> None:
    """Each advanced exact result must explain its work instead of using one placeholder."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult), case_id
    assert expected_result in format_ascii(result), case_id
    assert result.steps, case_id
    assert all(step.rule != _OPAQUE_RULE for step in result.steps), case_id
    assert all(step.explanation.strip() for step in result.steps), case_id
    assert all(step.verification.detail.strip() for step in result.steps), case_id
    assert any(step.notes for step in result.steps), case_id
    assert all(format_latex_expression(step.after) for step in result.steps), case_id
    payload = solve_payload(result)
    assert payload.status == "exact", case_id
    assert all(step.rule != _OPAQUE_RULE for step in payload.steps), case_id


def test_integral_without_a_specific_template_keeps_a_verified_answer(
    solver: Solver,
) -> None:
    """A verified antiderivative must not be discarded just because no template matched."""
    result = solver.solve("integrate(1/(x^4+1),x)")
    assert isinstance(result, ExactResult)
    assert result.steps
    assert result.steps[0].rule == "Verify the antiderivative by differentiation"
    assert result.steps[0].verification.method is VerificationMethod.DIFFERENTIATION
    assert _OPAQUE_RULE not in format_ascii(result)


def test_matrix_route_uses_actual_row_operations(solver: Solver) -> None:
    """Row reduction should expose elementary operations instead of one backend jump."""
    result = solver.solve("rref(matrix([[0,1],[1,0]]))")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules == ("Swap two rows", "Read the reduced row echelon form")
    assert "Result: [matrix([[1, 0], [0, 1]]), [0, 1]]" in format_ascii(result)
    assert result.steps[0].notes[0].label == "Rows swapped"


def test_inverse_route_checks_the_adjugate_formula_and_identity(solver: Solver) -> None:
    """An inverse derivation should verify both the formula and A * A^-1 = I."""
    result = solver.solve("inverse(matrix([[1,2],[3,4]]))")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Check that the matrix is invertible",
        "Form the adjugate over the determinant",
        "Divide every entry by the determinant",
        "Check the inverse by multiplication",
    )
    assert any(note.label == "Determinant" for note in result.steps[0].notes)
    assert result.steps[-1].verification.method.value == "substitution"


def test_ode_route_checks_the_solution_family_by_substitution(solver: Solver) -> None:
    """An ODE answer should name its order and independently verify the residual."""
    result = solver.solve("dsolve(diff(y(x),x)=y(x),y(x))")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Identify the differential equation",
        "Apply the classified solution method",
        "Check the solution by substitution",
    )
    assert result.steps[-1].verification.method.value == "substitution"
    assert any(note.label == "Arbitrary constants" for note in result.steps[-1].notes)


def test_j01_native_first_order_ode_input_is_supported(solver: Solver) -> None:
    """The benchmark J01 query should work without an adapter override."""
    result = solver.solve("dsolve(diff(y(x), x) = 3*y(x), y(x))")
    assert isinstance(result, ExactResult)
    assert "Result: Eq(y(x), C1 * exp(3 * x))" in format_ascii(result)
    assert tuple(step.rule for step in result.steps) == (
        "Identify the differential equation",
        "Apply the classified solution method",
        "Check the solution by substitution",
    )


def test_j02_native_second_order_ode_input_is_supported(solver: Solver) -> None:
    """The benchmark J02 query should work without an adapter override."""
    result = solver.solve("dsolve(diff(y(x), x, 2) + y(x) = 0, y(x))")
    assert isinstance(result, ExactResult)
    assert "Result: Eq(y(x), C1 * sin(x) + C2 * cos(x))" in format_ascii(result)
    assert result.steps[-1].verification.method.value == "substitution"


def test_inequality_route_shows_boundaries_and_interval_tests(solver: Solver) -> None:
    """A quadratic inequality should show the sign-chart method and its test points."""
    result = solver.solve("solve_inequality(x^2<4,x)")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules == (
        "Move every term to one side",
        "Find the boundary points",
        "Test one point in each interval",
    )
    assert result.steps[1].notes
    assert result.steps[2].notes
    assert "Result: (-2 < x) & (x < 2)" in format_ascii(result)


def test_ordinary_supported_paths_are_not_routed_into_the_guard(solver: Solver) -> None:
    """Normal algebra and calculus queries must keep their direct derivations."""
    for query, rule in (
        ("integrate(x^2,x)", "Use the power rule"),
        ("diff(sin(x),x)", "Differentiate the sine"),
        ("solve(x^2-4=0,x)", "Factor the quadratic"),
    ):
        result = solver.solve(query)
        assert isinstance(result, ExactResult), query
        assert result.steps, query
        assert result.steps[0].rule == rule, query


def test_symbolic_row_reduction_without_a_specific_template_keeps_a_verified_answer(
    solver: Solver,
) -> None:
    """A verified symbolic matrix result must not be discarded for lack of a template."""
    result = solver.solve("rref(matrix([[x,1],[1,0]]))")
    assert isinstance(result, ExactResult)
    assert result.steps
    assert result.steps[0].rule == "State the verified matrix result"
    assert _OPAQUE_RULE not in format_ascii(result)


def test_one_by_one_determinant_uses_its_only_entry(solver: Solver) -> None:
    """A 1 by 1 determinant should not be routed through a larger cofactor formula."""
    result = solver.solve("det(matrix([[5]]))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Determinant of a 1 by 1 matrix",)
    assert "Result: 5" in format_ascii(result)


def test_three_by_three_determinant_expands_along_the_first_row(solver: Solver) -> None:
    """A larger determinant should expose cofactors and their signed evaluation."""
    result = solver.solve("det(matrix([[1,2,3],[0,1,4],[5,6,0]]))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Expand along the first row",
        "Evaluate the smaller determinants",
    )
    assert "Result: 1" in format_ascii(result)
    assert len(result.steps[0].notes) == _THREE_BY_THREE_MINORS


def test_row_reduction_scales_and_eliminates_in_both_directions(solver: Solver) -> None:
    """Gauss-Jordan steps must include scaling and elimination above and below pivots."""
    result = solver.solve("rref(matrix([[2,4],[1,3]]))")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules[0] == "Scale a row to make its pivot 1"
    assert rules.count("Eliminate an entry in the pivot column") == _TWO_ELIMINATIONS
    assert "Result: [matrix([[1, 0], [0, 1]]), [0, 1]]" in format_ascii(result)


def test_zero_matrix_has_zero_rank_and_no_row_operations(solver: Solver) -> None:
    """A zero matrix should be handled without inventing a nonzero pivot."""
    rank = solver.solve("rank(matrix([[0,0],[0,0]]))")
    assert isinstance(rank, ExactResult)
    assert tuple(step.rule for step in rank.steps) == ("Count the pivot rows",)
    assert "Result: 0" in format_ascii(rank)
    reduced = solver.solve("rref(matrix([[0,0],[0,0]]))")
    assert isinstance(reduced, ExactResult)
    assert tuple(step.rule for step in reduced.steps) == ("Read the reduced row echelon form",)
    assert "Result: [matrix([[0, 0], [0, 0]]), []]" in format_ascii(reduced)


def test_repeated_eigenvalues_report_their_multiplicity(solver: Solver) -> None:
    """A repeated characteristic root must be counted, not silently deduplicated."""
    result = solver.solve("eigenvalues(matrix([[2,1,0],[0,2,0],[0,0,3]]))")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Form the characteristic equation",
        "Factor the characteristic polynomial",
        "Read the eigenvalues and their multiplicities",
    )
    assert "Result: {2: 2, 3: 1}" in format_ascii(result)
    assert any(note.label == "Multiplicity of 2" for note in result.steps[-1].notes)


def test_non_strict_inequality_keeps_its_boundary_points(solver: Solver) -> None:
    """A less-than-or-equal inequality must include the tested boundary roots."""
    result = solver.solve("solve_inequality(x^2<=4,x)")
    assert isinstance(result, ExactResult)
    assert "Result: (-2 <= x) & (x <= 2)" in format_ascii(result)
    assert "together with the zeros that are not excluded by a pole" in result.steps[-1].explanation


@pytest.mark.parametrize(
    ("query", "expected_rule"),
    [
        ("solve_inequality(x^2+1<0,x)", "Move every term to one side"),
        ("solve_inequality(sin(x)<0,x)", "State the verified solution set"),
        ("dsolve(y(x)=0,y(x))", "State the verified solution family"),
        ("rsolve(a(n)=0,a(n))", "State the verified recurrence solution"),
    ],
)
def test_advanced_operations_keep_verified_answers_with_honest_steps(
    solver: Solver,
    query: str,
    expected_rule: str,
) -> None:
    """A verified backend result is retained with an honest, non-opaque step."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult), query
    assert result.steps, query
    assert result.steps[0].rule == expected_rule, query
    assert _OPAQUE_RULE not in format_ascii(result), query
