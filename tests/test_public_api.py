"""Executable checks for the library examples documented in the README."""

from stepsolver import (
    ExactResult,
    Operation,
    Solver,
    SymbolicBackend,
    SympyBackend,
    format_expression,
    parse,
    solve_payload,
)


def test_domain_result_exposes_complete_solution_steps() -> None:
    """Library users should receive the same structured derivation shown on the web."""
    result = Solver().solve("integrate(x*exp(x), x)")
    assert isinstance(result, ExactResult)
    assert result.steps
    step = result.steps[0]
    assert step.rule
    assert step.explanation
    assert format_expression(step.before)
    assert format_expression(step.after)
    assert step.notes
    assert step.verification.detail


def test_presentation_adapter_renders_domain_steps_as_latex() -> None:
    """The presentation adapter should preserve steps while adding rendered forms."""
    result = Solver().solve("integrate(x*exp(x), x)")
    payload = solve_payload(result)
    assert payload.status == "exact"
    assert payload.result_latex is not None
    assert len(payload.steps) == len(result.steps)
    assert payload.steps[0].rule == result.steps[0].rule
    assert payload.steps[0].explanation == result.steps[0].explanation
    assert payload.steps[0].before_latex
    assert payload.steps[0].after_latex
    assert payload.as_dict()["steps"]


def test_parser_and_backend_can_be_composed_explicitly() -> None:
    """Parsing, orchestration, and symbolic execution should remain separate layers."""
    query = parse("limit(sin(x)/x, x, 0)")
    backend: SymbolicBackend = SympyBackend()
    result = Solver(backend=backend).solve(query)
    assert query.operation is Operation.LIMIT
    assert isinstance(result, ExactResult)
