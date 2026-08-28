"""Security and protocol regressions for the disposable worker entry point."""

import json

import pytest

from stepsolver import LimitError, parse_isolated_request, solve_isolated
from stepsolver.isolated import IsolatedLimits
from stepsolver.isolated_runner import run


def test_isolated_runner_returns_one_typed_bounded_solution() -> None:
    """A valid request should expose normalized math and worked steps."""
    response = json.loads(run(b'{"expression":"solve(x^2-4=0,x)"}'))

    assert response["ok"] is True
    solution = response["solution"]
    assert solution["normalized_expression"] == "solve(x ^ 2 - 4 = 0, x)"
    assert solution["status"] == "exact"
    assert solution["result"] == "[{x: -2}, {x: 2}]"
    assert solution["latex"] == r"x = -2\quad\text{or}\quad x = 2"
    assert solution["steps"]


@pytest.mark.parametrize(
    ("document", "code"),
    [
        (b'{"expression":"2+2","extra":true}', "invalid_request"),
        (b'{"expression":2}', "invalid_request"),
        ('{"expression":"π"}'.encode(), "invalid_request"),
    ],
)
def test_isolated_runner_rejects_non_contract_requests(document: bytes, code: str) -> None:
    """The process boundary should fail closed without a traceback."""
    response = json.loads(run(document))
    assert response == {
        "error": {"code": code, "message": "The expression is invalid."},
        "ok": False,
    }


@pytest.mark.parametrize(
    "expression",
    [
        "2^101",
        "sum(x,x,1,10001)",
        "det([[1,2,3],[4,5,6],[7,8,9]])",
    ],
)
def test_isolated_limits_are_calculated_before_symbolic_execution(expression: str) -> None:
    """Typed AST limits should reject expensive shapes before creating a solver."""
    limits = IsolatedLimits(matrix_rows=2, matrix_columns=2)
    with pytest.raises(LimitError):
        parse_isolated_request(json.dumps({"expression": expression}).encode(), limits)


def test_isolated_result_respects_a_serialized_byte_ceiling() -> None:
    """The response ceiling should be enforced after typed presentation."""
    request = parse_isolated_request(b'{"expression":"integrate(x*exp(x),x)"}')
    with pytest.raises(LimitError, match="serialized result"):
        solve_isolated(request, limits=IsolatedLimits(response_bytes=100))


@pytest.mark.parametrize(
    ("document", "limits", "code"),
    [
        (b'{"expression":"2+2"}', IsolatedLimits(request_bytes=8), "input_limit"),
        (b'{"expression":"2^101"}', IsolatedLimits(), "expression_limit"),
        (b'{"expression":"2+2"}', IsolatedLimits(response_bytes=32), "result_limit"),
    ],
)
def test_isolated_runner_reports_the_limit_layer(
    document: bytes,
    limits: IsolatedLimits,
    code: str,
) -> None:
    """The supervisor should distinguish transport, expression, and result bounds."""
    response = json.loads(run(document, limits))

    assert response["ok"] is False
    assert response["error"]["code"] == code
