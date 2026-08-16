"""Web frontend and API tests."""

import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from stepsolver import web
from stepsolver.browser import solve_mathjson_json
from stepsolver.web import SolveResponse, create_app

_HTTP_OK = 200
_HTTP_UNPROCESSABLE_CONTENT = 422
_CONTOUR_STEP_COUNT = 2
_MINIMUM_THREE_COLUMN_GRID_COUNT = 2
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
        runtime = client.get("/static/runtime.mjs")
        favicon = client.get("/static/favicon.svg")
    assert homepage.status_code == _HTTP_OK
    assert "StepSolver" in homepage.text
    assert "<title>∬ StepSolver</title>" in homepage.text
    assert '<span class="wordmark-symbol" aria-hidden="true">∬</span>' in homepage.text
    assert "math-field" in homepage.text
    assert "Mathematical symbol palette" in homepage.text
    assert "<select" not in homepage.text
    assert 'class="site-header"' in homepage.text
    assert 'href="./static/favicon.svg"' in homepage.text
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
    assert 'from "./vendor.mjs"' in script.text
    assert 'from "./runtime.mjs"' in script.text
    assert 'new URL("./fonts/", import.meta.url).href' in script.text
    assert runtime.status_code == _HTTP_OK
    assert favicon.status_code == _HTTP_OK
    assert "∬" in favicon.text
    assert 'fetch("./api/solve"' in runtime.text
    assert "firstElementChild" not in script.text
    assert "expressionField.insert" in script.text
    assert 'key.addEventListener("pointerdown"' in script.text
    assert "event.preventDefault()" in script.text
    assert "event.detail === 0" in script.text
    assert "insertSymbolTemplate(key)" in script.text
    assert "expressionField.executeCommand" not in script.text
    assert 'behavior: "smooth"' not in script.text
    assert (
        stylesheet.text.count("grid-template-columns: repeat(3, minmax(0, 1fr))")
        >= _MINIMUM_THREE_COLUMN_GRID_COUNT
    )
    assert ".math-toolbar" in stylesheet.text
    assert "overflow: hidden" in stylesheet.text
    assert "touch-action: manipulation" in stylesheet.text
    assert "pointer-events: none" in stylesheet.text
    assert "--smart-fence-color: var(--ink)" in stylesheet.text
    assert 'table.className = "derivative-table"' in script.text
    assert "Differentiate each factor once" in script.text


def test_frontend_handles_large_math_and_warms_the_browser_runtime() -> None:
    """Large formulas should scroll, while browser startup happens outside the main thread."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
        runtime = client.get("/static/runtime.mjs")
    assert 'class="math-viewport result-viewport"' in homepage.text
    assert 'aria-label="Scrollable answer"' in homepage.text
    assert "createMathViewport" in script.text
    assert "solverClient.warmup()" in script.text
    assert '"requestIdleCallback" in window' in script.text
    assert "querySelector('[part=\"keyboard-sink\"]')" in script.text
    assert 'keyboardSink.setAttribute("inputmode", "text")' in script.text
    assert "window.mathVirtualKeyboard?.hide()" in script.text
    assert "async warmup()" in runtime.text
    assert ".math-viewport" in stylesheet.text
    assert ".primary-math-field::part(virtual-keyboard-toggle)" in stylesheet.text
    assert ".primary-math-field::part(keyboard-sink)" in stylesheet.text
    assert "overscroll-behavior-inline: contain" in stylesheet.text
    assert "touch-action: pan-x" in stylesheet.text
    assert "pointer-events: none" in stylesheet.text
    assert "user-select: none" in stylesheet.text
    assert "-webkit-overflow-scrolling: touch" in stylesheet.text


def test_browser_runtime_serializes_the_same_solver_payload() -> None:
    """The Pyodide boundary should expose real answers and worked steps as JSON."""
    source = json.dumps(
        ["Integrate", ["Sin", "x"], ["Tuple", "x", 0, "Pi"]],
    )
    payload = SolveResponse.model_validate_json(solve_mathjson_json(source))
    assert payload.status == "exact"
    assert payload.result_latex == "2"
    assert payload.steps
    assert payload.steps[0].before_latex.startswith(r"\int_{0}^{\pi}")


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


def test_divergent_absolute_value_integral_has_a_graphical_proof() -> None:
    """The browser should distinguish a proved divergence from an unsolved query."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\int_0^\infty |x|\,\mathrm{d}x",
                "math_json": [
                    "Integrate",
                    ["Abs", "x"],
                    ["Tuple", "x", 0, "PositiveInfinity"],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert payload.status == "divergent"
    assert payload.result_latex == r"\text{Diverges to }+\infty"
    assert payload.reason == "The improper integral diverges to +infinity."
    assert [step.rule for step in payload.steps] == [
        "Use the sign of x on the interval",
        "Rewrite the improper integral as a limit",
        "Apply the Fundamental Theorem of Calculus",
        "Evaluate the finite bounds",
        "Test the endpoint limit",
    ]
    assert payload.steps[0].before_latex == r"\int_{0}^{\infty} \left|x\right|\,\mathrm{d}x"
    assert payload.steps[-1].after_latex == r"\infty"
    assert "Integral(" not in payload.formatted_ascii


def test_visual_adjacency_uses_the_product_rule() -> None:
    """Adjacent visual factors should reach the solver as an ordinary product."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\frac{\mathrm{d}}{\mathrm{d}x}(\sin(x)e^x)",
                "math_json": [
                    "D",
                    [
                        "InvisibleOperator",
                        ["Sin", "x"],
                        ["Power", "ExponentialE", "x"],
                    ],
                    "x",
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert payload.status == "exact"
    assert [step.rule for step in payload.steps] == ["Apply the product rule"]
    assert payload.result_latex == (
        r"\exp\left(x\right) \cdot \sin\left(x\right) + "
        r"\exp\left(x\right) \cdot \cos\left(x\right)"
    )
    assert payload.steps[0].notes[0].label == "Product rule"
    assert "log" not in payload.result_latex
    assert "InvisibleOperator" not in response.text


def test_visual_limit_tuple_is_accepted() -> None:
    """Compute Engine's bound-variable tuple should decode as a limit query."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\lim_{x\to0}\frac{\sin(x)}{x}",
                "math_json": [
                    "Limit",
                    ["Divide", ["Sin", "x"], "x"],
                    ["Tuple", "x", 0],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert payload.status == "exact"
    assert payload.result_latex == "1"
    assert [step.rule for step in payload.steps] == ["Use the standard sine limit"]


def test_reciprocal_quadratic_integral_contains_detailed_latex_steps() -> None:
    """The browser should receive the completed-square integration derivation."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\int\frac{1}{x^2-x+1}\,\mathrm{d}x",
                "math_json": [
                    "Integrate",
                    [
                        "Divide",
                        1,
                        ["Add", ["Subtract", ["Power", "x", 2], "x"], 1],
                    ],
                    "x",
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert [step.rule for step in payload.steps] == [
        "Complete the square",
        "Substitute to get a unit denominator",
        "Use the basic arctangent rule",
        "Substitute back",
    ]
    assert payload.steps[0].after_latex.startswith(r"\int \frac{1}")
    assert [note.label for note in payload.steps[0].notes] == [
        "Take half the linear coefficient, then square it",
        "Add and subtract that number",
        "Recognize the perfect square",
        "General pattern",
    ]
    assert payload.steps[0].notes[0].expression_latex == (
        r"\left(\frac{-1}{2}\right)^{2} = \frac{1}{4}"
    )
    assert [note.label for note in payload.steps[1].notes] == [
        "Choose the substitution",
        "Rewrite the shifted term",
        "Change the differential",
    ]
    assert payload.steps[1].after_latex == (
        r"\frac{2}{\sqrt{3}} \cdot \int \frac{1}{u^{2} + 1}\,\mathrm{d}u"
    )
    assert payload.steps[1].notes[0].expression_latex == (r"u = \frac{2 \cdot x - 1}{\sqrt{3}}")
    assert payload.steps[1].notes[2].expression_latex == (
        r"\mathrm{d}x = \frac{\sqrt{3}}{2} \cdot \mathrm{d}u"
    )
    assert payload.steps[2].notes[0].expression_latex == (
        r"\int \frac{1}{u^{2} + 1}\,\mathrm{d}u = \arctan\left(u\right) + C"
    )
    assert r"\frac{1}{\frac{\sqrt{3}}{2}}" not in payload.steps[2].after_latex
    assert r"\arctan" in payload.steps[3].after_latex
    assert payload.result_latex is not None
    assert payload.result_latex.endswith("+ C")


def test_dirichlet_integral_contains_human_parameter_steps() -> None:
    """The browser should receive the full damping-parameter derivation."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\int_0^\infty\frac{\sin(x)}{x}\,\mathrm{d}x",
                "math_json": [
                    "Integrate",
                    ["Divide", ["Sin", "x"], "x"],
                    ["Tuple", "x", 0, "PositiveInfinity"],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert response.status_code == _HTTP_OK
    assert [step.rule for step in payload.steps] == [
        "Introduce a damping parameter",
        "Differentiate with respect to the parameter",
        "Recover the parameterized integral",
        "Determine the constant",
        "Remove the damping",
    ]
    assert payload.steps[0].after_latex == r"\lim_{a \to 0^{+}} F\left(a\right)"
    assert payload.steps[1].after_latex == (
        r"\frac{\mathrm{d}}{\mathrm{d}a}\left(F\left(a\right)\right) "
        r"= \frac{-1}{a^{2} + 1}"
    )
    assert payload.steps[3].notes[2].expression_latex == r"C = \frac{\pi}{2}"
    assert payload.steps[-1].after_latex == r"\frac{\pi}{2}"


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
