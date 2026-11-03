"""Derivation-family regressions for absolute, logarithmic, exponential, series, and limits."""

import pytest

from stepsolver import (
    DivergenceKind,
    DivergentResult,
    ExactResult,
    NoSolutionValue,
    Solver,
    SolveResult,
    UndefinedResult,
    format_ascii,
    format_expression,
)


@pytest.fixture
def solver() -> Solver:
    """Create the symbolic solver once per regression."""
    return Solver()


def _rules(result: SolveResult) -> tuple[str, ...]:
    return tuple(step.rule for step in result.steps)


@pytest.mark.parametrize(
    ("query", "expected_values", "check_rule"),
    [
        ("solve(abs(x - 1) = 2, x)", ("-1", "3"), "Split into the two absolute-value branches"),
        ("solve(abs(2*x + 3) = 5, x)", ("-4", "1"), "Check candidates in the original equation"),
        ("solve(abs(x) = 3, x)", ("-3", "3"), "Solve both sign branches"),
        ("solve(abs(x) = -1, x)", (), "Check candidates in the original equation"),
    ],
)
def test_absolute_value_equations_show_both_branches(
    solver: Solver,
    query: str,
    expected_values: tuple[str, ...],
    check_rule: str,
) -> None:
    """Absolute-value equations must split into sign branches and verify candidates."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert check_rule in rules
    rendered = format_ascii(result)
    for value in expected_values:
        assert value in rendered
    assert "Compute exact result" not in rules


def test_absolute_value_equation_with_no_solution_is_explicit(solver: Solver) -> None:
    """An absolute value can never equal a negative number."""
    result = solver.solve("solve(abs(x) = -1, x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert _rules(result)[-1] == "Check candidates in the original equation"


@pytest.mark.parametrize(
    ("query", "first_rule", "last_rule"),
    [
        (
            "solve(ln(x) = 2, x)",
            "Exponentiate both sides",
            "Check candidates in the logarithm's domain",
        ),
        (
            "solve(log(x, 10) = 2, x)",
            "Exponentiate both sides",
            "Check candidates in the logarithm's domain",
        ),
        (
            "solve(log(x) + log(x - 3) = 1, x)",
            "Combine the logarithms",
            "Check candidates in the logarithm's domain",
        ),
        (
            "solve(log(x, 2) = log(x - 2, 2) + 1, x)",
            "Move the logarithms to one side",
            "Check candidates in the logarithm's domain",
        ),
    ],
)
def test_logarithmic_equations_expose_domain_checks(
    solver: Solver,
    query: str,
    first_rule: str,
    last_rule: str,
) -> None:
    """Logarithmic equations must undo the logarithm and keep only valid arguments."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert rules[0] == first_rule
    assert rules[-1] == last_rule
    constraints = {
        format_expression(constraint.expression)
        for step in result.steps
        for constraint in step.introduced_constraints
    }
    assert any("> 0" in constraint for constraint in constraints)


def test_logarithmic_equation_with_constant_combines_the_quotient(solver: Solver) -> None:
    """Moving a constant first keeps the logarithm laws visible."""
    result = solver.solve("solve(log(x, 2) = log(x - 2, 2) + 1, x)")
    assert isinstance(result, ExactResult)
    assert _rules(result)[:2] == (
        "Move the logarithms to one side",
        "Combine the logarithms",
    )
    assert "Result: [{x: 4}]" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "expected_rules", "rendered"),
    [
        (
            "solve(exp(x) = 5, x)",
            ("Take logarithms of both sides", "Solve the resulting equation"),
            "log(5)",
        ),
        (
            "solve(2^x = 8, x)",
            ("Write both sides with the same base", "Solve the resulting equation"),
            "3",
        ),
        (
            "solve(exp(2*x) - 3*exp(x) + 2 = 0, x)",
            ("Substitute u = exp(x)", "Solve the polynomial in u"),
            "log(2)",
        ),
        (
            "solve(4^x - 5*2^x + 4 = 0, x)",
            ("Substitute u = 2^x", "Solve the polynomial in u"),
            "2",
        ),
    ],
)
def test_exponential_equations_use_their_structural_method(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
    rendered: str,
) -> None:
    """Exponential equations must show the inverse operation or a substitution."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    for rule in expected_rules:
        assert rule in rules
    assert rendered in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "formula_rule", "term_count"),
    [
        ("series(exp(x), x, 0, 4)", "Write the Maclaurin series formula", 4),
        ("series(sin(x), x, 0, 5)", "Write the Maclaurin series formula", 2),
        ("series(log(1 + x), x, 0, 5)", "Write the Maclaurin series formula", 4),
        ("series(exp(x), x, 1, 4)", "Write the Taylor series about x = 1", 4),
    ],
)
def test_series_expansions_show_derivative_values(
    solver: Solver,
    query: str,
    formula_rule: str,
    term_count: int,
) -> None:
    """Taylor and Maclaurin series must be assembled from derivative values."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert rules[0] == formula_rule
    assert rules[-1] == "Assemble the truncated series"
    term_rules = [rule for rule in rules if rule.startswith("Form the ")]
    assert len(term_rules) == term_count
    notes = [note.label for step in result.steps for note in step.notes]
    assert notes.count("Derivative value at the center") == term_count
    assert "O(" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "rendered"),
    [
        ("sum(1/2^n, n, 0, oo)", "2"),
        ("sum((1/3)^(2*n), n, 0, oo)", "9/8"),
        ("sum((1/2)^(3*n), n, 0, oo)", "8/7"),
        ("sum(3*((1/4)^n), n, 0, oo)", "4"),
        ("sum((1/2)^n, n, 2, oo)", "1/2"),
    ],
)
def test_geometric_series_identity_handles_amplitude_and_slope(
    solver: Solver,
    query: str,
    rendered: str,
) -> None:
    """A geometric series may have an outside factor or a scaled exponent."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert _rules(result) == ("Apply the infinite geometric-series identity",)
    step = result.steps[0]
    labels = {note.label for note in step.notes}
    assert {"Identity", "Required condition", "Common ratio"} <= labels
    assert rendered in format_ascii(result)


def test_geometric_series_convergence_condition_is_shown_for_divergence(
    solver: Solver,
) -> None:
    """A common ratio outside the unit interval must fail the convergence test."""
    result = solver.solve("sum(2^n, n, 0, oo)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.POSITIVE_INFINITY
    assert _rules(result) == ("Apply the geometric-series convergence test",)


@pytest.mark.parametrize("query", ["limit(1/x,x,0)", "limit(abs(x)/x,x,0)"])
def test_two_sided_limit_with_disagreeing_sides_does_not_exist(
    solver: Solver,
    query: str,
) -> None:
    """A two-sided limit must fail when the one-sided limits differ."""
    result = solver.solve(query)
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.NONFINITE
    assert _rules(result) == (
        "Evaluate the left-hand limit",
        "Evaluate the right-hand limit",
        "Conclude the two-sided limit does not exist",
    )
    rendered = format_ascii(result)
    assert "undefined()" in rendered
    assert "zoo" not in rendered
    assert "nan" not in rendered


def test_oscillating_limit_is_reported_as_nonexistent(solver: Solver) -> None:
    """An oscillation without a limit must not leak an accumulation-bounds object."""
    result = solver.solve("limit(sin(1/x),x,0)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.NONFINITE
    rendered = format_ascii(result)
    assert "AccumBounds" not in rendered


def test_one_sided_infinite_limit_remains_an_exact_value(solver: Solver) -> None:
    """A requested one-sided infinity is a valid exact answer."""
    result = solver.solve("limit(1/x,x,0,right)")
    assert isinstance(result, ExactResult)
    assert format_ascii(result).endswith("Result: oo")


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("integrate(1/x, x, -1, 1)", DivergenceKind.NONFINITE),
        ("integrate(1/x^2, x, -1, 1)", DivergenceKind.POSITIVE_INFINITY),
    ],
)
def test_interior_singularity_splits_the_improper_integral(
    solver: Solver,
    query: str,
    kind: DivergenceKind,
) -> None:
    """An interior pole requires two independent one-sided limits."""
    result = solver.solve(query)
    assert isinstance(result, DivergentResult)
    assert result.kind is kind
    assert _rules(result) == (
        "Split the integral at the interior singularity",
        "Evaluate the left piece as a limit",
        "Evaluate the right piece as a limit",
        "Conclude the integral diverges",
    )
    rendered = format_ascii(result)
    assert "zoo" not in rendered
    assert "nan" not in rendered


def test_undefined_division_shows_simplification_before_the_conclusion(
    solver: Solver,
) -> None:
    """A zero denominator must be simplified before the undefined conclusion."""
    result = solver.solve("1/(x-x)")
    assert isinstance(result, UndefinedResult)
    assert _rules(result) == (
        "Simplify the denominator",
        "Conclude the expression is undefined",
    )
    assert format_expression(result.steps[0].after) == "0"
    assert format_expression(result.steps[-1].after) == "undefined"


@pytest.mark.parametrize(
    ("query", "expected_rendered"),
    [
        ("solve([x^2 + y = 5, x - y = 1], [x, y])", "-3"),
        ("solve([x^2 + y^2 = 25, y - x = 1], [x, y])", "3"),
        ("solve([x^2 + y^2 = 1, x + y = 2], [x, y])", "No solution"),
    ],
)
def test_nonlinear_two_by_two_system_uses_substitution(
    solver: Solver,
    query: str,
    expected_rendered: str,
) -> None:
    """A nonlinear two-by-two system must isolate, substitute, and check pairs."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert rules[0].startswith("Solve the first equation for ")
    assert "Substitute into the other equation" in rules
    assert "Compute exact result" not in rules
    assert expected_rendered in format_ascii(result)


def test_equal_logarithms_equate_arguments_and_check_the_domain(solver: Solver) -> None:
    """Equal logarithms reduce to equal arguments, then every candidate is checked."""
    result = solver.solve("solve(log(x) = log(2*x - 3), x)")
    assert isinstance(result, ExactResult)
    assert _rules(result)[0] == "Equate the logarithms' arguments"
    assert _rules(result)[-1] == "Check candidates in the logarithm's domain"
    assert "Result: [{x: 3}]" in format_ascii(result)


def test_equal_logarithms_reject_a_candidate_outside_the_domain(solver: Solver) -> None:
    """A solution of the argument equation may violate a logarithm's positivity."""
    result = solver.solve("solve(log(x) = log(2*x), x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert _rules(result)[-1] == "Check candidates in the logarithm's domain"


def test_equal_exponentials_equate_the_exponents(solver: Solver) -> None:
    """The natural exponential is one-to-one, so equal values imply equal exponents."""
    result = solver.solve("solve(exp(x) = exp(2*x - 1), x)")
    assert isinstance(result, ExactResult)
    assert _rules(result)[0] == "Equate the exponents"
    assert "Result: [{x: 1}]" in format_ascii(result)


def test_logarithm_difference_combines_into_a_quotient(solver: Solver) -> None:
    """A difference of same-base logarithms must be combined before exponentiating."""
    result = solver.solve("solve(log(x) - log(x - 1) = 1, x)")
    assert isinstance(result, ExactResult)
    assert _rules(result)[0] == "Combine the logarithms"
    labels = {note.label for step in result.steps for note in step.notes}
    assert "Logarithm law" in labels
    assert _rules(result)[-1] == "Check candidates in the logarithm's domain"
    assert "Result: [{x: E / (-1 + E)}]" in format_ascii(result)


def test_exponential_substitution_rejects_the_nonpositive_root(solver: Solver) -> None:
    """A zero root of the polynomial in u is impossible for an exponential."""
    result = solver.solve("solve(exp(2*x) - exp(x) = 0, x)")
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Substitute u = exp(x)",
        "Solve the polynomial in u",
        "Reject nonpositive values of u",
        "Recover the variable from the substitution",
    )
    assert "Result: [{x: 0}]" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "rendered"),
    [
        ("sum((1/2)^n, n, 0, 5)", "47 / 32"),
        ("sum(3*2^n, n, 1, 4)", "90"),
    ],
)
def test_finite_geometric_series_uses_the_closed_form(
    solver: Solver,
    query: str,
    rendered: str,
) -> None:
    """A finite geometric sum must substitute both bounds into the closed form."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert _rules(result) == ("Apply the finite geometric-series identity",)
    labels = {note.label for note in result.steps[0].notes}
    assert {"Identity", "Required condition", "Common ratio"} <= labels
    condition = next(
        note.expression for note in result.steps[0].notes if note.label == "Required condition"
    )
    assert format_expression(condition).endswith("!= 1")
    assert rendered in format_ascii(result)


def test_convergent_interior_singularity_adds_both_pieces(solver: Solver) -> None:
    """A singularity whose one-sided pieces are finite still needs the split."""
    result = solver.solve("integrate(log(x^2), x, -1, 1)")
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Split the integral at the interior singularity",
        "Evaluate the left piece as a limit",
        "Evaluate the right piece as a limit",
        "Add the two one-sided values",
    )
    assert format_ascii(result).endswith("Result: -4")


def test_three_by_three_linear_system_eliminates_variables_in_order(solver: Solver) -> None:
    """A consistent three-by-three system must show forward and back substitution."""
    result = solver.solve("solve([x + y + z = 6, 2*x - y + z = 3, x + 2*y - z = 2], [x, y, z])")
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Eliminate x from equation 2",
        "Eliminate x from equation 3",
        "Eliminate y from equation 3",
        "Solve for z",
        "Substitute back for y",
        "Substitute back for x",
    )
    assert "Result: [{x: 1, y: 2, z: 3}]" in format_ascii(result)


def test_dependent_three_by_three_system_reports_a_solution_family(solver: Solver) -> None:
    """A free variable must appear explicitly in the parametric solution."""
    result = solver.solve("solve([x + y + z = 6, 2*x + 2*y + 2*z = 12, x - y = 1], [x, y, z])")
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert "Recognize a dependent equation" in rules
    assert rules[-1] == "Write the solution family"
    rendered = format_ascii(result)
    assert "7 / 2 - z / 2" in rendered
    assert "5 / 2 - z / 2" in rendered


def test_inconsistent_three_by_three_system_is_explicit(solver: Solver) -> None:
    """A contradictory row must end with an inconsistency conclusion."""
    result = solver.solve("solve([x + y + z = 1, x + y + z = 2, x - y = 0], [x, y, z])")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert _rules(result)[-1] == "Conclude the system is inconsistent"
    assert format_ascii(result).endswith("Result: No solution")


def test_nonlinear_system_without_a_real_root_is_explicit(solver: Solver) -> None:
    """Substitution may reduce to an equation with no real root."""
    result = solver.solve("solve([exp(x) + y = 1, x + y = 1], [x, y])")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert "Conclude the system has no solution" in _rules(result)
    assert format_ascii(result).endswith("Result: No solution")


@pytest.mark.parametrize(
    "query",
    [
        "solve(abs(2*x - 1) = 3, x)",
        "solve(abs(x^2 - 1) = 0, x)",
    ],
)
def test_absolute_value_equations_handle_nearby_structures(
    solver: Solver,
    query: str,
) -> None:
    """Coefficient and quadratic variations must still split and verify."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Split into the two absolute-value branches",
        "Solve both sign branches",
        "Check candidates in the original equation",
    )
    assert "Compute exact result" not in format_ascii(result)


def test_three_by_three_system_moves_a_pivot_into_position(solver: Solver) -> None:
    """A leading equation without the pivot variable must be reordered."""
    result = solver.solve("solve([z = 1, x + y + z = 3, x - y = 1], [x, y, z])")
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert rules[0] == "Move an equation with x into position"
    assert "Move an equation with y into position" in rules
    assert rules[-1] == "Substitute back for x"


def test_three_by_three_system_without_a_leading_x_term(solver: Solver) -> None:
    """Every equation may omit the first pivot variable except through others."""
    result = solver.solve("solve([x + z = 3, y + z = 4, x + y = 5], [x, y, z])")
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Eliminate x from equation 3",
        "Eliminate y from equation 3",
        "Solve for z",
        "Substitute back for y",
        "Substitute back for x",
    )


def test_nonlinear_system_with_a_repeated_root(solver: Solver) -> None:
    """A repeated candidate must collapse to one verified pair."""
    result = solver.solve("solve([x*y = 1, x + y = 2], [x, y])")
    assert isinstance(result, ExactResult)
    assert _rules(result) == (
        "Solve the first equation for x",
        "Substitute into the other equation",
        "Solve the resulting equation for y",
        "Back-substitute to find the other variable",
    )
    assert format_ascii(result).count("Result: [{") == 1


def test_nonlinear_system_with_two_quadratic_branches(solver: Solver) -> None:
    """A quadratic reduction may produce two real candidate values."""
    result = solver.solve("solve([x^2 - y = 0, y - x - 2 = 0], [x, y])")
    assert isinstance(result, ExactResult)
    rules = _rules(result)
    assert rules[1] == "Substitute into the other equation"
    assert rules[-1] == "Back-substitute to find the other variable"
    rendered = format_ascii(result)
    assert "2" in rendered
    assert "-1" in rendered
