"""End-to-end symbolic solver tests."""

import pytest

from stepsolver import (
    DivergenceKind,
    DivergentResult,
    ExactResult,
    NoSolutionValue,
    Solver,
    UndefinedResult,
    UnsolvedResult,
    format_ascii,
    format_latex_expression,
)

_CONTOUR_STEP_COUNT = 2
_CROSSED_NUMERIC_FACTOR_COUNT = 4
_INTRODUCED_MULTIPLIER_COUNT = 2


@pytest.fixture
def solver() -> Solver:
    """Create the default symbolic solver."""
    return Solver()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("2+2", "Result: 4"),
        ("simplify((x^2-1)/(x-1))", "Result: x + 1"),
        ("expand((x+1)^3)", "Result: x ^ 3 + 3 * x ^ 2 + 3 * x + 1"),
        ("factor(x^2-4)", "Result: (x - 2) * (x + 2)"),
        ("diff(sin(x)*exp(x),x)", "Result: exp(x) * sin(x) + exp(x) * cos(x)"),
        ("integrate(sin(x),x,0,pi)", "Result: 2"),
        ("limit(sin(x)/x,x,0)", "Result: 1"),
        ("sum(k,k,1,10)", "Result: 55"),
        ("product(k,k,1,5)", "Result: 120"),
        ("det(matrix([[1,2],[3,4]]))", "Result: -2"),
        ("rank(matrix([[1,2],[2,4]]))", "Result: 1"),
        ("gcd(84,30)", "Result: 6"),
        ("lcm(12,18)", "Result: 36"),
        ("is_prime(104729)", "Result: true"),
        ("prime_factors(360)", "Result: {2: 3, 3: 2, 5: 1}"),
        ("binomial(10,3)", "Result: 120"),
        ("permutations(6,2)", "Result: 30"),
        ("combinations(6,2)", "Result: 15"),
    ],
)
def test_supported_queries_have_exact_results(
    solver: Solver,
    query: str,
    expected: str,
) -> None:
    """Representative operations should produce typed exact values and steps."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert result.steps
    assert expected in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "index", "denominator"),
    [
        ("sum(1/n^6,n,0,oo)", "n \\ne 0", "n^{6} \\ne 0"),
        ("sum(1/(n-2),n,0,4)", "n \\ne 2", "n - 2 \\ne 0"),
        (
            "sum(1/(n*(n-1)),n,1,5)",
            "n \\ne 1",
            "n \\cdot \\left(n - 1\\right) \\ne 0",
        ),
    ],
)
def test_sums_with_included_singular_terms_are_undefined(
    solver: Solver,
    query: str,
    index: str,
    denominator: str,
) -> None:
    """A pole at an included index must never leak SymPy's `zoo` sentinel."""
    result = solver.solve(query)
    assert isinstance(result, UndefinedResult)
    rendered = format_ascii(result)
    assert "zoo" not in rendered
    assert "denominator is zero" in result.reason
    constraints = result.steps[0].introduced_constraints
    assert tuple(format_latex_expression(item.expression) for item in constraints) == (
        denominator,
        index,
    )


def test_pole_outside_sum_range_does_not_make_series_undefined(solver: Solver) -> None:
    """Only singular terms actually included in the index range invalidate a sum."""
    result = solver.solve("sum(1/(n-2),n,3,oo)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.POSITIVE_INFINITY
    assert "series" in result.reason
    assert tuple(step.rule for step in result.steps) == (
        "Shift the summation index",
        "Apply the p-series test",
    )


def test_shifted_convergent_p_series_reduces_to_the_standard_identity(
    solver: Solver,
) -> None:
    """An index shift should expose the standard p-series instead of hiding the method."""
    result = solver.solve("sum(1/(n-2)^2,n,3,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Shift the summation index",
        "Recognize a convergent p-series",
        "Use the exact value of zeta(2)",
    )
    assert result.steps[0].notes[0].label == "Index substitution"
    assert format_ascii(result).endswith("Result: pi ^ 2 / 6")


def test_constant_zero_denominator_is_undefined_for_every_term(solver: Solver) -> None:
    """A summand that divides by zero everywhere must not be called divergent."""
    result = solver.solve("sum(1/0,n,0,3)")
    assert isinstance(result, UndefinedResult)
    rendered = format_ascii(result)
    assert "zero denominator at every index" in result.steps[0].explanation
    assert "1 / 0" in rendered
    assert "zoo" not in rendered


@pytest.mark.parametrize(
    ("query", "rules", "answer"),
    [
        (
            "sum(1/n^6,n,1,oo)",
            ("Recognize a convergent p-series", "Use the exact value of zeta(6)"),
            "Result: pi ^ 6 / 945",
        ),
        (
            "sum(1/n^3,n,1,oo)",
            ("Recognize a convergent p-series",),
            "Result: zeta(3)",
        ),
        (
            "sum(n^2,n,1,10)",
            ("Use the sum of squares identity", "Simplify the arithmetic"),
            "Result: 385",
        ),
        (
            "sum((1/2)^n,n,0,oo)",
            ("Apply the infinite geometric-series identity",),
            "Result: 2",
        ),
        (
            "sum((1/2)^n,n,1,oo)",
            ("Apply the infinite geometric-series identity",),
            "Result: 1",
        ),
    ],
)
def test_common_sums_show_human_first_identities(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
    answer: str,
) -> None:
    """Common sums should state the reusable identity before giving the answer."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert result.steps[0].notes
    assert all(step.before != step.after for step in result.steps)
    assert format_ascii(result).endswith(answer)


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("sum(sin(n*pi/2)/n,n,1,oo)", "Result: pi / 4"),
        ("sum(sin(3*n*pi/2)/n,n,1,oo)", "Result: -pi / 4"),
        ("sum(3*sin(n*pi/3)/n,n,1,oo)", "Result: pi"),
        ("sum(sin(5*n*pi/2)/n,n,1,oo)", "Result: pi / 4"),
        ("sum(sin(sqrt(2)*n)/n,n,1,oo)", "Result: -sqrt(2) / 2 + pi / 2"),
    ],
)
def test_harmonic_sine_series_use_the_general_abel_derivation(
    solver: Solver,
    query: str,
    answer: str,
) -> None:
    """Exact real angles should use one periodic Fourier rule rather than value lookups."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Introduce an Abel convergence factor",
        "Sum the damped sine series",
        "Remove the convergence factor",
        "Simplify the exact value",
    )
    assert "Dirichlet's test proves convergence" in result.steps[0].explanation
    assert result.steps[1].notes[0].label == "Damped sine-series identity"
    assert result.steps[2].notes[0].label == "Harmonic sine-series identity"
    assert format_ascii(result).endswith(answer)


def test_full_turn_harmonic_sine_series_simplifies_every_term(solver: Solver) -> None:
    """Angles congruent to zero should take the elementary zero-term path."""
    result = solver.solve("sum(sin(2*n*pi)/n,n,1,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == ("Simplify the sine terms",)
    assert format_ascii(result).endswith("Result: 0")


@pytest.mark.parametrize(
    ("query", "identity_rule", "answer"),
    [
        (
            "sum((-1)^(n+1)/n,n,1,oo)",
            "Use the alternating harmonic-series identity",
            "Result: log(2)",
        ),
        (
            "sum((-1)^(n+1)/n^2,n,1,oo)",
            "Use the Dirichlet eta identity",
            "Result: pi ^ 2 / 12",
        ),
        (
            "sum(3*(-1)^n/n^2,n,1,oo)",
            "Use the Dirichlet eta identity",
            "Result: -(pi ^ 2) / 4",
        ),
    ],
)
def test_alternating_p_series_apply_leibniz_before_the_exact_identity(
    solver: Solver,
    query: str,
    identity_rule: str,
    answer: str,
) -> None:
    """Alternating exact values should first establish convergence by Leibniz's test."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Apply the Leibniz alternating-series test",
        identity_rule,
    )
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Core magnitude limit",
        "Decreasing core magnitudes",
    )
    assert format_ascii(result).endswith(answer)


def test_nonzero_term_limit_uses_the_nth_term_divergence_test(solver: Solver) -> None:
    """A series whose terms do not vanish should fail before stronger tests are attempted."""
    result = solver.solve("sum((n+1)/n,n,1,oo)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.POSITIVE_INFINITY
    assert tuple(step.rule for step in result.steps) == ("Apply the nth-term divergence test",)
    assert result.steps[0].notes[0].label == "Term limit"


@pytest.mark.parametrize("base", [2, 3, 5])
def test_indexed_root_power_series_has_human_divergence_steps(
    solver: Solver,
    base: int,
) -> None:
    """Matching indexed roots should simplify visibly before the term test."""
    result = solver.solve(f"sum(root({base}^n,n)+{base}^n,n,1,oo)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.POSITIVE_INFINITY
    assert tuple(step.rule for step in result.steps) == (
        "Simplify the indexed root",
        "Apply the nth-term divergence test",
    )
    assert format_latex_expression(result.steps[0].before) == (
        rf"\sum_{{n=1}}^{{\infty}} \left(\sqrt[n]{{{base}^{{n}}}} + {base}^{{n}}\right)"
    )
    assert format_latex_expression(result.steps[0].after) == (
        rf"\sum_{{n=1}}^{{\infty}} \left({base} + {base}^{{n}}\right)"
    )
    assert format_latex_expression(result.steps[0].notes[0].expression) == (
        rf"\sqrt[n]{{{base}^{{n}}}} = {base}"
    )
    assert result.steps[1].notes[0].label == "Term limit"


@pytest.mark.parametrize(
    ("query", "kind", "rule"),
    [
        ("sum(1/n,n,1,oo)", DivergenceKind.POSITIVE_INFINITY, "Apply the p-series test"),
        (
            "sum(2^n,n,0,oo)",
            DivergenceKind.POSITIVE_INFINITY,
            "Apply the geometric-series convergence test",
        ),
        (
            "sum((-1)^n,n,0,oo)",
            DivergenceKind.NONFINITE,
            "Apply the geometric-series convergence test",
        ),
        (
            "sum((-1)^n,n,1,oo)",
            DivergenceKind.NONFINITE,
            "Apply the geometric-series convergence test",
        ),
    ],
)
def test_divergent_series_are_classified_with_a_human_test(
    solver: Solver,
    query: str,
    kind: DivergenceKind,
    rule: str,
) -> None:
    """Known divergent families should not become exact or unevaluated answers."""
    result = solver.solve(query)
    assert isinstance(result, DivergentResult)
    assert result.kind is kind
    assert result.steps[0].rule == rule
    assert "series" in result.reason
    assert "Sum(" not in format_ascii(result)


def test_equation_system_returns_typed_mappings(solver: Solver) -> None:
    """Equation systems should return structured solution mappings."""
    result = solver.solve("solve([x+y=3,x-y=1],[x,y])")
    assert isinstance(result, ExactResult)
    assert "Result: [{x: 2, y: 1}]" in format_ascii(result)


def test_linear_system_has_elimination_and_substitution_steps(solver: Solver) -> None:
    """A two-by-two system should be solved the way a student would solve it."""
    result = solver.solve("solve([x+y=3,x-y=1],[x,y])")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Eliminate x",
        "Solve for y",
        "Substitute back to find x",
    )
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Scaled first equation",
        "Scaled second equation",
    )


def test_inconsistent_system_is_no_solution(solver: Solver) -> None:
    """A contradictory elimination row should produce a typed empty solution set."""
    result = solver.solve("solve([x+y=1,x+y=2],[x,y])")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert result.steps[-1].rule == "Conclude the system is inconsistent"
    assert format_ascii(result).endswith("Result: No solution")


def test_dependent_system_explains_its_free_variable(solver: Solver) -> None:
    """A dependent system should explain its solution family instead of invoking the backend."""
    result = solver.solve("solve([x+y=2,2*x+2*y=4],[x,y])")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Eliminate x",
        "Write the solution family",
    )
    assert "free" in result.steps[-1].explanation


def test_three_by_three_system_uses_elimination_and_back_substitution(
    solver: Solver,
) -> None:
    """A three-variable system should expose the same algebra a student writes by hand."""
    result = solver.solve("solve([x+y+z=6,x-y+z=2,2*x+y-z=1],[x,y,z])")
    assert isinstance(result, ExactResult)
    rules = tuple(step.rule for step in result.steps)
    assert rules[:3] == (
        "Eliminate x from equation 2",
        "Eliminate x from equation 3",
        "Eliminate y from equation 3",
    )
    assert rules[-3:] == ("Solve for z", "Solve for y", "Substitute back for x")
    assert all(step.rule != "Compute exact result" for step in result.steps)


@pytest.mark.parametrize(
    ("query", "expected_rules"),
    [
        (
            "solve([y=2,x+y=3],[x,y])",
            ("Use the equation without x", "Substitute back to find x"),
        ),
        (
            "solve([x+y+z=3,2*x+2*y+2*z=6,x-y=0],[x,y,z])",
            (
                "Recognize a dependent equation",
                "Eliminate x from equation 3",
                "Move an equation with y into position",
                "Write the solution family",
            ),
        ),
        (
            "solve([x+y=2,x-y=0,2*x=2],[x,y])",
            (
                "Eliminate x from equation 2",
                "Eliminate x from equation 3",
                "Recognize a dependent equation",
                "Solve for y",
                "Substitute back for x",
            ),
        ),
    ],
)
def test_linear_system_edge_cases_keep_student_facing_steps(
    solver: Solver,
    query: str,
    expected_rules: tuple[str, ...],
) -> None:
    """Zero pivots, dependent rows, and extra equations should remain worked solutions."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert all(step.rule != "Compute exact result" for step in result.steps)


def test_system_derivations_use_braces_and_readable_row_operations(solver: Solver) -> None:
    """System steps should look like systems, not backend lists or raw row symbols."""
    result = solver.solve("solve([x+y+z=6,x-y+z=2,2*x+y-z=1],[x,y,z])")
    assert isinstance(result, ExactResult)
    first_step_latex = format_latex_expression(result.steps[0].before)
    row_operation_latex = format_latex_expression(result.steps[0].notes[0].expression)
    assert first_step_latex.startswith(r"\begin{cases}")
    assert first_step_latex.endswith(r"\end{cases}")
    assert row_operation_latex == r"R_{2} \leftarrow R_{2} - R_{1}"


def test_two_by_two_elimination_uses_smallest_integer_multipliers(solver: Solver) -> None:
    """Elimination should avoid needlessly doubling both scaled equations."""
    result = solver.solve("solve([2*x+3*y=13,4*x-y=5],[x,y])")
    assert isinstance(result, ExactResult)
    scaled = tuple(format_latex_expression(note.expression) for note in result.steps[0].notes)
    assert scaled == (
        r"4 \cdot x + 6 \cdot y = 26",
        r"-4 \cdot x + y = -5",
    )


@pytest.mark.parametrize(
    ("query", "rules"),
    [
        (
            "solve(2*x+3=7,x)",
            (
                "Subtract the constant from both sides",
                "Cancel the opposite constants",
                "Simplify both sides",
                "Divide both sides by the coefficient",
                "Cancel the common coefficient",
                "Simplify the quotients",
            ),
        ),
        (
            "solve(x^2-4=0,x)",
            (
                "Factor the quadratic",
                "Apply the zero-product property",
                "Solve each factor",
            ),
        ),
        (
            "solve(x^2+x-1=0,x)",
            ("Apply the quadratic formula",),
        ),
        (
            "solve((x+1)/(x-2)=3,x)",
            (
                "Multiply both sides by the denominator",
                "Cancel the common factors",
                "Expand both sides",
                "Subtract the variable term from both sides",
                "Group like terms",
                "Combine the variable terms",
                "Add the constant to both sides",
                "Cancel the opposite constants",
                "Simplify both sides",
                "Divide both sides by the coefficient",
                "Cancel the common coefficient",
                "Simplify the quotients",
                "Write the variable on the left",
            ),
        ),
    ],
)
def test_equations_have_verified_detailed_derivations(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
) -> None:
    """Common equation families should expose real algebraic transformations."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert all(step.before != step.after for step in result.steps)
    assert all(
        step.verification.method.value == "solution-set equivalence" for step in result.steps
    )


def test_fractional_linear_equation_clears_the_actual_denominator(
    solver: Solver,
) -> None:
    """Numeric fractions should multiply by their LCD without fake domain restrictions."""
    result = solver.solve("solve(5*x/2+2=3*x/2+10,x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Multiply both sides by the denominator",
        "Cancel the common factors",
        "Subtract the variable term from both sides",
        "Group like terms",
        "Combine the variable terms",
        "Subtract the constant from both sides",
        "Cancel the opposite constants",
        "Simplify both sides",
        "Divide both sides by the coefficient",
        "Cancel the common coefficient",
        "Simplify the quotients",
    )
    assert result.steps[0].introduced_constraints == ()
    assert format_latex_expression(result.steps[0].after) == (
        r"\textcolor{#4f8cff}{2 \cdot \left(\textcolor{#f4f4f5}"
        r"{\frac{5 \cdot x}{2} + 2}\right)}"
        r" = \textcolor{#4f8cff}{2 \cdot \left(\textcolor{#f4f4f5}"
        r"{\frac{3 \cdot x}{2} + 10}\right)}"
    )
    cancellation = format_latex_expression(result.steps[1].before)
    canceled_two = r"\textcolor{#ffffff}{\cancel{\textcolor{#ff5362}{2}}}"
    assert cancellation.count(canceled_two) == _CROSSED_NUMERIC_FACTOR_COUNT
    assert r"\xcancel{" not in cancellation
    assert format_latex_expression(result.steps[1].after) == r"5 \cdot x + 4 = 3 \cdot x + 20"
    assert format_latex_expression(result.steps[2].before) == (
        format_latex_expression(result.steps[1].after)
    )
    assert format_latex_expression(result.steps[2].after) == (
        r"5 \cdot x + 4 \textcolor{#4f8cff}{- 3 \cdot x}"
        r" = 3 \cdot x + 20 \textcolor{#4f8cff}{- 3 \cdot x}"
    )
    grouped = format_latex_expression(result.steps[3].after)
    assert grouped.startswith(r"\left(5 \cdot x - 3 \cdot x\right) + 4")
    assert grouped.count(r"\cancel{\textcolor{#ff5362}{3 \cdot x}}") == _INTRODUCED_MULTIPLIER_COUNT
    constant_operation = format_latex_expression(result.steps[5].after)
    assert constant_operation.count(r"\textcolor{#4f8cff}{- 4}") == _INTRODUCED_MULTIPLIER_COUNT
    assert format_latex_expression(result.steps[7].after) == r"2 \cdot x = 16"
    division = format_latex_expression(result.steps[8].after)
    assert division.count(r"\textcolor{#4f8cff}{\frac{") == _INTRODUCED_MULTIPLIER_COUNT
    coefficient_cancellation = format_latex_expression(result.steps[9].after)
    assert coefficient_cancellation.count(canceled_two) == _INTRODUCED_MULTIPLIER_COUNT
    assert format_latex_expression(result.steps[10].after) == "x = 8"


@pytest.mark.parametrize(
    ("query", "multiplier", "expected_rules"),
    [
        (
            "solve(x/2+1=x/3+2,x)",
            "6",
            (
                "Multiply both sides by the denominator",
                "Cancel the common factors",
                "Subtract the variable term from both sides",
                "Group like terms",
                "Combine the variable terms",
                "Subtract the constant from both sides",
                "Cancel the opposite constants",
                "Simplify both sides",
            ),
        ),
        (
            "solve(5*x/3-1=x/3+7,x)",
            "3",
            (
                "Multiply both sides by the denominator",
                "Cancel the common factors",
                "Subtract the variable term from both sides",
                "Group like terms",
                "Combine the variable terms",
                "Add the constant to both sides",
                "Cancel the opposite constants",
                "Simplify both sides",
                "Divide both sides by the coefficient",
                "Cancel the common coefficient",
                "Simplify the quotients",
            ),
        ),
    ],
)
def test_nearby_numeric_fraction_equations_use_their_lcd(
    solver: Solver,
    query: str,
    multiplier: str,
    expected_rules: tuple[str, ...],
) -> None:
    """Nearby fractional linear equations should retain the same classroom method."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == expected_rules
    assert result.steps[0].introduced_constraints == ()
    multiplied = format_latex_expression(result.steps[0].after)
    highlighted_product = rf"\textcolor{{#4f8cff}}{{{multiplier} \cdot \left("
    assert multiplied.startswith(highlighted_product)
    assert multiplied.count(highlighted_product) == _INTRODUCED_MULTIPLIER_COUNT
    cancellation = format_latex_expression(result.steps[1].before)
    assert r"\cancel" in cancellation
    assert r"\xcancel" not in cancellation
    assert r"\textcolor{#ff5362}" in cancellation
    assert r"\textcolor{#ffffff}" in cancellation


def test_exponential_linear_equation_uses_lambert_w_steps(solver: Solver) -> None:
    """A standard Lambert W equation should show the substitutions that produce its answer."""
    result = solver.solve("solve(exp(x)+x=0,x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Isolate the exponential",
        "Substitute the negative variable",
        "Create the Lambert W pattern",
        "Apply the Lambert W function",
        "Substitute back",
    )
    assert all(step.rule != "Compute exact result" for step in result.steps)
    assert result.steps[3].notes[0].label == "Lambert W identity"


@pytest.mark.parametrize("query", ["diff(x^4,x,2)", "diff(x^3,x,2)"])
def test_higher_derivatives_repeat_the_human_rule(solver: Solver, query: str) -> None:
    """Higher derivatives should show each differentiation instead of a backend jump."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Use the power rule",
        "Use the power rule",
    )
    assert all(step.rule != "Compute exact result" for step in result.steps)


def test_rational_equation_reducing_to_cubic_has_human_steps(solver: Solver) -> None:
    """An irreducible rational cubic should use a practical human-first method."""
    result = solver.solve("solve(x^2-4=1/(x+1),x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Multiply both sides by the denominator",
        "Cancel the common factors",
        "Expand and collect like terms",
        "Approximate the real root",
    )
    assert tuple(
        format_latex_expression(item.expression) for item in result.steps[0].introduced_constraints
    ) == (r"x + 1 \ne 0", r"x \ne -1")
    assert format_latex_expression(result.steps[0].after) == (
        r"\textcolor{#4f8cff}{\left(x + 1\right) \cdot \left("
        r"\textcolor{#f4f4f5}{x^{2} - 4}\right)}"
        r" = \textcolor{#4f8cff}{\left(x + 1\right) \cdot \left("
        r"\textcolor{#f4f4f5}{\frac{1}{x + 1}}\right)}"
    )
    assert format_latex_expression(result.steps[1].before) == (
        r"\left(x + 1\right) \cdot \left(x^{2} - 4\right)"
        r" = \frac{\textcolor{#ffffff}{\cancel{\textcolor{#ff5362}{x + 1}}}}"
        r"{\textcolor{#ffffff}{\cancel{\textcolor{#ff5362}{x + 1}}}}"
    )
    assert format_latex_expression(result.steps[1].after) == (
        r"\left(x + 1\right) \cdot \left(x^{2} - 4\right) = 1"
    )
    assert tuple(note.label for note in result.steps[3].notes) == (
        "Rational-root test",
        "Bracket the root",
        "Newton iteration",
        "Successive estimates",
        "Exact form (optional)",
    )
    assert format_latex_expression(result.steps[3].notes[0].expression) == (
        r"\left[f\left(-5\right) = -85, f\left(-1\right) = -1, "
        r"f\left(1\right) = -7, f\left(5\right) = 125\right]"
    )
    assert format_latex_expression(result.steps[3].notes[1].expression) == (
        r"\left[f\left(2\right) = -1, f\left(3\right) = 19\right]"
    )
    assert format_latex_expression(result.steps[3].notes[2].expression) == (
        r"x_{k+1} = x_k - \frac{f\left(x_k\right)}{f'\left(x_k\right)}"
    )
    assert format_latex_expression(result.steps[3].after) == r"x \approx 2.079596"
    assert format_latex_expression(result.steps[3].notes[-1].expression).startswith(
        r"x = \frac{-1}{3} + \sqrt[3]"
    )
    assert "sqrt[3]" not in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "rules"),
    [
        (
            "solve(x^3-1=0,x)",
            (
                "Factor the polynomial",
                "Apply the zero-product property",
                "Keep the real solutions",
            ),
        ),
        (
            "solve(x^4-5*x^2+4=0,x)",
            (
                "Factor the polynomial",
                "Apply the zero-product property",
                "Keep the real solutions",
            ),
        ),
        ("solve(x^5-x+1=0,x)", ("Approximate the real roots",)),
        (
            "solve(x^3=1/(x+1),x)",
            (
                "Multiply both sides by the denominator",
                "Cancel the common factors",
                "Expand and collect like terms",
                "Approximate the real roots",
            ),
        ),
        ("solve(x^4+x+1=0,x)", ("Check for real roots",)),
    ],
)
def test_higher_degree_polynomials_do_not_fall_back_to_answer_only(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
) -> None:
    """Polynomial equations should expose a real solving strategy beyond degree two."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert all(step.rule != "Compute exact result" for step in result.steps)
    if result.steps[-1].rule == "Approximate the real roots":
        assert tuple(note.label for note in result.steps[-1].notes) == ("Newton iteration",)


def test_reciprocal_quadratic_integral_has_worked_steps(solver: Solver) -> None:
    """A completed-square integral should expose its arctangent derivation."""
    result = solver.solve("integrate(1/(x^2-x+1),x)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Complete the square",
        "Substitute to get a unit denominator",
        "Use the basic arctangent rule",
        "Substitute back",
    )
    assert all(step.before != step.after for step in result.steps)
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Take half the linear coefficient, then square it",
        "Add and subtract that number",
        "Recognize the perfect square",
        "General pattern",
    )
    assert result.steps[1].verification.method.value == "substitution"
    assert tuple(note.label for note in result.steps[1].notes) == (
        "Choose the substitution",
        "Rewrite the shifted term",
        "Change the differential",
    )
    assert result.steps[2].verification.method.value == "differentiation"
    assert tuple(note.label for note in result.steps[2].notes) == ("Rule to remember",)
    assert "u²" not in result.steps[2].explanation
    assert format_ascii(result).endswith("+ C")


def test_dirichlet_integral_has_a_parameter_derivation(solver: Solver) -> None:
    """The classical improper integral should explain the damping-parameter method."""
    result = solver.solve("integrate(sin(x)/x,x,0,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    )
    assert result.steps[1].verification.method.value == "differentiation"
    assert tuple(note.label for note in result.steps[3].notes) == (
        "The damped integral vanishes",
        "Arctangent limit",
        "Therefore",
    )
    assert format_ascii(result).endswith("pi / 2")


@pytest.mark.parametrize(
    ("query", "rules"),
    [
        (
            "solve((x-1)^2=0,x)",
            ("Set the repeated factor equal to zero", "Solve each factor"),
        ),
        (
            "solve(2*x^2+4*x+2=0,x)",
            (
                "Factor the quadratic",
                "Set the repeated factor equal to zero",
                "Solve each factor",
            ),
        ),
        (
            "solve(x^2+1=0,x)",
            ("Calculate the discriminant", "Conclude there are no real solutions"),
        ),
        (
            "solve(x=x+1,x)",
            ("Simplify the equation", "Conclude there are no solutions"),
        ),
        ("integrate(sin(x),x)", ("Use the sine antiderivative",)),
        (
            "integrate(sin(2*x),x)",
            ("Apply the reverse chain rule for sine",),
        ),
        ("integrate(x^2,x)", ("Use the power rule",)),
        (
            "integrate(x/(x^2+1),x)",
            ("Substitute the denominator", "Use the logarithm rule", "Substitute back"),
        ),
        ("integrate(1/(x^2+1),x)", ("Use the basic arctangent rule",)),
        (
            "integrate(1/(4*x^2+1),x)",
            (
                "Substitute to get a unit denominator",
                "Use the basic arctangent rule",
                "Substitute back",
            ),
        ),
    ],
)
def test_human_first_edge_case_derivations(
    solver: Solver,
    query: str,
    rules: tuple[str, ...],
) -> None:
    """Degenerate and direct cases should use the shortest meaningful method."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == rules
    assert all(step.rule != "Compute exact result" for step in result.steps)


def test_derivations_hide_neutral_algebra_artifacts(solver: Solver) -> None:
    """Student-facing transformations should not display identity arithmetic."""
    direct = solver.solve("integrate(1/(x^2+1),x)")
    scaled = solver.solve("integrate(1/(4*x^2+1),x)")
    logarithmic = solver.solve("integrate(x/(x^2+1),x)")
    assert isinstance(direct, ExactResult)
    assert isinstance(scaled, ExactResult)
    assert isinstance(logarithmic, ExactResult)
    rendered = " ".join(
        format_latex_expression(expression)
        for result in (direct, scaled, logarithmic)
        for step in result.steps
        for expression in (step.before, step.after)
    )
    assert "1 \\cdot" not in rendered
    assert r"\frac{2}{2}" not in rendered
    assert r"\frac{\frac{1}{2}}{1}" not in rendered
    positivity = logarithmic.steps[1].notes[0].expression
    assert format_latex_expression(positivity) == r"x^{2} + 1 > 0"


@pytest.mark.parametrize("query", ["solve(x=x,x)", "solve((x-1)/(x-1)=1,x)"])
def test_identity_equations_do_not_return_a_false_empty_set(
    solver: Solver,
    query: str,
) -> None:
    """Universal identities should remain honest until domain sets are representable."""
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert "identity" not in result.reason.lower() or "domain" in result.reason.lower()
    assert "universal solution set" in result.reason


def test_no_real_solution_is_rendered_as_the_empty_set(solver: Solver) -> None:
    """A negative discriminant should end with mathematical empty-set notation."""
    result = solver.solve("solve(x^2+1=0,x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    assert format_latex_expression(result.steps[-1].after) == r"\varnothing"


def test_denominator_restrictions_are_explicit_and_filter_candidates(
    solver: Solver,
) -> None:
    """Clearing a denominator must record and enforce the original domain."""
    result = solver.solve("solve((x-1)/(x-1)=x,x)")
    assert isinstance(result, ExactResult)
    assert isinstance(result.value, NoSolutionValue)
    restrictions = result.steps[0].introduced_constraints
    assert tuple(format_latex_expression(item.expression) for item in restrictions) == (
        r"x - 1 \ne 0",
        r"x \ne 1",
    )
    assert result.steps[-1].rule == "Apply the domain restriction"


def test_each_original_denominator_adds_an_explicit_restriction(solver: Solver) -> None:
    """A rational equation should retain each excluded value without redundant products."""
    result = solver.solve("solve(1/(x-1)+1/(x+1)=0,x)")
    assert isinstance(result, ExactResult)
    restrictions = result.steps[0].introduced_constraints
    assert tuple(format_latex_expression(item.expression) for item in restrictions) == (
        r"x - 1 \ne 0",
        r"x + 1 \ne 0",
        r"x \ne 1",
        r"x \ne -1",
    )


def test_negative_power_introduces_a_denominator_restriction(solver: Solver) -> None:
    """Writing a reciprocal as a negative power must preserve the same domain."""
    result = solver.solve("solve((x-1)^-1=1,x)")
    assert isinstance(result, ExactResult)
    assert tuple(
        format_latex_expression(item.expression) for item in result.steps[0].introduced_constraints
    ) == (r"x - 1 \ne 0", r"x \ne 1")


def test_scaled_dirichlet_integral_reduces_to_the_standard_case(solver: Solver) -> None:
    """A positive sine frequency should be removed by an explicit scaling substitution."""
    result = solver.solve("integrate(sin(2*x)/x,x,0,oo)")
    assert isinstance(result, ExactResult)
    assert tuple(step.rule for step in result.steps) == (
        "Scale the integration variable",
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    )
    assert tuple(note.label for note in result.steps[0].notes) == (
        "Choose the substitution",
        "Rewrite the original variable",
        "Change the differential",
    )
    assert format_ascii(result).endswith("pi / 2")


def test_inequality_solver(solver: Solver) -> None:
    """A real univariate inequality should produce an exact interval condition."""
    result = solver.solve("solve_inequality(x^2<4,x)")
    assert isinstance(result, ExactResult)
    rendered = format_ascii(result)
    assert "-2 < x" in rendered
    assert "x < 2" in rendered


def test_parameterized_unit_circle_integral(solver: Solver) -> None:
    """The positively oriented unit circle integral of 1/z should be 2*pi*i."""
    result = solver.solve("contour_integrate(1/z,z,exp(i*t),t,0,2*pi)")
    assert isinstance(result, ExactResult)
    assert len(result.steps) == _CONTOUR_STEP_COUNT
    assert "Result: 2 * i * pi" in format_ascii(result)


def test_reversed_contour_reverses_sign(solver: Solver) -> None:
    """Reversing the unit-circle parameter bounds should reverse orientation."""
    result = solver.solve("contour_integrate(1/z,z,exp(i*t),t,2*pi,0)")
    assert isinstance(result, ExactResult)
    assert "Result: -2 * i * pi" in format_ascii(result)


def test_matrix_inverse_is_structured(solver: Solver) -> None:
    """Matrix-valued answers should remain structured StepSolver values."""
    result = solver.solve("inverse(matrix([[1,2],[3,4]]))")
    assert isinstance(result, ExactResult)
    assert "Result: matrix([[-2, 1], [3/2, -1/2]])" in format_ascii(result)


def test_numeric_is_explicit(solver: Solver) -> None:
    """Numerical approximation should require the numeric operation."""
    result = solver.solve("numeric(pi,8)")
    assert isinstance(result, ExactResult)
    assert "Result: 3.1415927" in format_ascii(result)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cancel((x^2-1)/(x-1))", "Result: x + 1"),
        ("apart(1/(x*(x+1)),x)", "Result: -1 / (x + 1) + 1 / x"),
        ("diff(x^4,x,2)", "Result: 12 * x ^ 2"),
        ("diff(log(x^2+1),x)", "Result: 2 * x / (x ^ 2 + 1)"),
        ("integrate(x^2,x)", "Result: x ^ 3 / 3"),
        ("limit(1/x,x,0,right)", "Result: oo"),
        ("series(exp(x),x,0,4)", "Result: 1 + x + x ^ 2 / 2 + x ^ 3 / 6 + O(x ^ 4)"),
        ("matrix([[1,2],[3,4]])", "Result: matrix([[1, 2], [3, 4]])"),
        ("rref(matrix([[1,2],[2,4]]))", "Result: [matrix([[1, 2], [0, 0]]), [0]]"),
        ("eigenvalues(matrix([[2,0],[0,3]]))", "Result: {2: 1, 3: 1}"),
        ("dsolve(diff(y(x),x)=y(x),y(x))", "Result: Eq(y(x), C1 * exp(x))"),
        ("rsolve(a(n+1)-a(n)=1,a(n))", "Result: C0 + n"),
        ("laplace(exp(-t),t,s)", "Result: 1 / (s + 1)"),
        ("inverse_laplace(1/(s+1),s,t)", "Result: exp(-t) * Heaviside(t)"),
        ("fourier(exp(-pi*x^2),x,k)", "Result: exp(-pi * k ^ 2)"),
        ("inverse_fourier(exp(-pi*k^2),k,x)", "Result: exp(-pi * x ^ 2)"),
        ("numeric(sqrt(2))", "Result: 1.41421356237310"),
    ],
)
def test_extended_symbolic_operations(solver: Solver, query: str, expected: str) -> None:
    """The broader calculus, matrix, transform, and recurrence surface should work."""
    result = solver.solve(query)
    assert isinstance(result, ExactResult)
    assert expected in format_ascii(result)


@pytest.mark.parametrize(
    "query",
    ["integrate(x)", "matrix([])", "is_prime(x)", "limit(x,x,0,sideways)"],
)
def test_invalid_operation_arguments_return_unsolved(solver: Solver, query: str) -> None:
    """Semantically invalid queries should return a typed partial result."""
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert result.reason


@pytest.mark.parametrize(
    ("query", "expected_first_rule"),
    [
        ("integrate(abs(x),x,0,oo)", "Use the sign of x on the interval"),
        ("integrate(abs(x),x,-oo,0)", "Use the sign of x on the interval"),
        ("integrate(abs(x),x,-oo,oo)", "Split the integral at zero"),
        ("integrate(x,x,0,oo)", "Rewrite the improper integral as a limit"),
    ],
)
def test_divergent_improper_integrals_are_explained_as_divergent(
    solver: Solver,
    query: str,
    expected_first_rule: str,
) -> None:
    """Divergent integrals should be conclusions, never fake exact values."""
    result = solver.solve(query)
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.POSITIVE_INFINITY
    assert result.steps[0].rule == expected_first_rule
    assert result.steps[-1].after != result.steps[-1].before
    assert all(step.rule != "Compute exact result" for step in result.steps)
    rendered = format_ascii(result)
    assert "Integral(" not in rendered
    assert rendered.endswith("The improper integral diverges to +infinity.")


def test_negative_infinite_improper_integral_has_its_own_direction(solver: Solver) -> None:
    """A negative divergent tail should retain the direction of divergence."""
    result = solver.solve("integrate(-x,x,0,oo)")
    assert isinstance(result, DivergentResult)
    assert result.kind is DivergenceKind.NEGATIVE_INFINITY
    assert result.steps[-1].verification.detail == (
        "The endpoint limit determines whether the improper integral converges."
    )


@pytest.mark.parametrize(
    "query",
    [
        "diff(f(x),x)",
        "integrate(f(x),x)",
        "sum(f(k),k,1,n)",
        "product(f(k),k,1,n)",
    ],
)
def test_unevaluated_backend_operations_are_never_exact(
    solver: Solver,
    query: str,
) -> None:
    """Backend placeholders must not leak through the exact-result boundary."""
    result = solver.solve(query)
    assert isinstance(result, UnsolvedResult)
    assert "could not evaluate" in result.reason
    assert "Compute exact result" not in format_ascii(result)
