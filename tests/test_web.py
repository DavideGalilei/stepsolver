"""Web frontend and API tests."""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from stepsolver import web
from stepsolver.web import SolveResponse, create_app

_HTTP_OK = 200
_HTTP_UNPROCESSABLE_CONTENT = 422
_CONTOUR_STEP_COUNT = 2
_TEST_HOST = "127.0.0.2"
_TEST_PORT = 8080


@dataclass(slots=True, kw_only=True)
class _UvicornCall:
    host: str = ""
    port: int = 0
    reload: bool = False


def test_homepage_and_assets_are_served() -> None:
    """The browser shell and bundled assets should be available."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
    assert homepage.status_code == _HTTP_OK
    assert "StepSolver" in homepage.text
    assert "math-field" in homepage.text
    assert "Mathematical symbol palette" in homepage.text
    assert "<select" not in homepage.text
    assert 'class="site-header"' in homepage.text
    assert "Try an example:" in homepage.text
    assert "section-number" not in homepage.text
    assert "status-pill" not in homepage.text
    assert stylesheet.status_code == _HTTP_OK
    assert "--accent" in stylesheet.text
    assert "radial-gradient" not in stylesheet.text
    assert "border-radius: 999" not in stylesheet.text
    assert "text-transform: uppercase" not in stylesheet.text
    assert script.status_code == _HTTP_OK
    assert "ComputeEngine" in script.text
    assert 'behavior: "smooth"' not in script.text


def test_exact_solve_response_contains_latex_steps() -> None:
    """Exact API responses should carry both ASCII and generated LaTeX."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\int_0^\pi\sin(x)\,\mathrm{d}x",
                "math_json": ["Integrate", ["Sin", "x"], ["Tuple", "x", 0, "Pi"]],
            },
        )
    assert response.status_code == _HTTP_OK
    payload = SolveResponse.model_validate_json(response.text)
    assert payload.status == "exact"
    assert payload.result_latex == "2"
    assert payload.steps
    assert payload.steps[0].before_latex.startswith(r"\int_{0}^{\pi}")


def test_graphical_equation_solve_response() -> None:
    """A visually entered equation should be solved without backend-call syntax."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": "x^2-4=0",
                "math_json": ["Equal", ["Subtract", ["Power", "x", 2], 4], 0],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert payload.status == "exact"
    assert payload.result_latex == (
        r"\left[\left\{x \mapsto -2\right\}, \left\{x \mapsto 2\right\}\right]"
    )
    assert [step.rule for step in payload.steps] == [
        "Factor the quadratic",
        "Apply the zero-product property",
        "Solve each factor",
    ]
    assert payload.steps[0].before_latex == "x^{2} - 4 = 0"
    assert payload.steps[-1].after_latex == r"\left[x = 2, x = -2\right]"
    assert all(step.verification_method == "solution-set equivalence" for step in payload.steps)
    assert all(r"\mathtt" not in step.after_latex for step in payload.steps)


def test_invalid_syntax_returns_validation_error() -> None:
    """Parser failures should become an HTTP 422 response."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={"latex": r"\frac{1}{}", "math_json": ["Error", "missing"]},
        )
    assert response.status_code == _HTTP_UNPROCESSABLE_CONTENT


def test_derivative_notation_is_inferred_without_an_operation_field() -> None:
    """Derivative notation should directly select differentiation."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\frac{\mathrm{d}}{\mathrm{d}x}x^2",
                "math_json": ["D", ["Power", "x", 2], "x"],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert payload.status == "exact"
    assert payload.steps[0].before_latex.startswith(r"\frac{\mathrm{d}}{\mathrm{d}x}")


def test_server_entry_point_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Poetry server command should honor documented environment settings."""
    call = _UvicornCall()

    def fake_serve(*, host: str, port: int, reload: bool) -> None:
        call.host = host
        call.port = port
        call.reload = reload

    monkeypatch.setenv("STEPSOLVER_HOST", _TEST_HOST)
    monkeypatch.setenv("STEPSOLVER_PORT", str(_TEST_PORT))
    monkeypatch.setenv("STEPSOLVER_RELOAD", "true")
    monkeypatch.setattr(web, "serve", fake_serve)
    web.main()

    assert call.host == _TEST_HOST
    assert call.port == _TEST_PORT
    assert call.reload


@pytest.mark.parametrize("port", ["invalid", "0", "65536"])
def test_server_entry_point_rejects_invalid_ports(
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    """Invalid server port settings should fail before starting Uvicorn."""
    monkeypatch.setenv("STEPSOLVER_PORT", port)
    with pytest.raises(ValueError, match="STEPSOLVER_PORT"):
        web.main()
