"""Contract tests between the pinned graphical editor and Python parser."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from stepsolver.browser import solve_mathjson_json
from stepsolver.mathjson import JsonValue
from stepsolver.web import SolveResponse

_FIXTURE_PATH = Path(__file__).parents[1] / "frontend" / "fixtures" / "editor-cases.json"


class _EditorCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    latex: str
    math_json: JsonValue
    result_latex: str
    expected_rules: tuple[str, ...]


_CASES = tuple(
    TypeAdapter(list[_EditorCase]).validate_json(_FIXTURE_PATH.read_text(encoding="utf-8"))
)


@pytest.mark.parametrize("editor_case", _CASES, ids=lambda item: item.name)
def test_editor_mathjson_reaches_the_browser_solver(editor_case: _EditorCase) -> None:
    """Every built-in editor example should solve through the browser JSON boundary."""
    response = solve_mathjson_json(json.dumps(editor_case.math_json))
    assert "Traceback" not in response
    payload = SolveResponse.model_validate_json(response)
    assert payload.status == "exact"
    assert payload.result_latex == editor_case.result_latex
    actual_rules = tuple(step.rule for step in payload.steps)
    assert actual_rules, (
        f"{editor_case.name} returned an exact answer without a derivation; "
        "human-solvable editor cases must include worked steps"
    )
    assert "Compute exact result" not in actual_rules, (
        f"{editor_case.name} fell back to an opaque exact-result step instead of its "
        f"required human method: {editor_case.expected_rules}"
    )
    assert actual_rules == editor_case.expected_rules


def test_browser_query_errors_do_not_expose_python_tracebacks() -> None:
    """Expected input errors should remain short and safe for display."""
    response = solve_mathjson_json(json.dumps(["Limit", "x", "x"]))
    payload: object = json.loads(response)
    assert payload == {"error": "limit approach must be a MathJSON function expression"}
    assert "Traceback" not in response
