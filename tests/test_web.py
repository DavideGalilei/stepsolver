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
_MINIMUM_DOCUMENT_OVERFLOW_GUARDS = 4
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
    assert "minimum-scale=1, shrink-to-fit=no, viewport-fit=cover" in homepage.text
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
    assert 'key.addEventListener("pointerdown"' not in script.text
    assert "insertSymbolTemplate(key)" in script.text
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


def test_desktop_palette_inserts_on_the_first_pointer_press() -> None:
    """Mouse focus transfer should not consume the first symbol-palette interaction."""
    with TestClient(create_app()) as client:
        script = client.get("/static/app.js")
    assert 'mathToolbar.addEventListener(\n  "pointerdown"' in script.text
    assert 'event.pointerType !== "mouse"' in script.text
    assert "event.button !== 0" in script.text
    assert "function symbolKeyFromEvent(event)" in script.text
    assert 'target.closest(".symbol-key")' in script.text
    assert "lastMouseInsertion?.key === key" in script.text
    assert "event.stopPropagation()" in script.text
    assert "{ capture: true }" in script.text


def test_header_has_a_github_star_link() -> None:
    """The header should link directly to the public repository."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
    assert 'class="github-star"' in homepage.text
    assert 'href="https://github.com/DavideGalilei/stepsolver"' in homepage.text
    assert 'aria-label="Star StepSolver on GitHub"' in homepage.text
    assert 'target="_blank" rel="noopener noreferrer"' in homepage.text
    assert "Star on GitHub" in homepage.text
    assert ".github-star" in stylesheet.text


def test_theme_toggle_honors_device_preference_and_persists_choice() -> None:
    """The theme should follow the OS until the visitor explicitly chooses one."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
    assert 'id="theme-toggle"' in homepage.text
    assert 'aria-pressed="false"' in homepage.text
    assert 'aria-label="Switch to dark theme"' in homepage.text
    assert 'localStorage.getItem("stepsolver-theme")' in homepage.text
    assert "document.documentElement.dataset.theme = theme" in homepage.text
    assert "@media (prefers-color-scheme: dark)" in stylesheet.text
    assert ':root:not([data-theme="light"])' in stylesheet.text
    assert ':root[data-theme="dark"]' in stylesheet.text
    assert ':root[data-theme="light"] { color-scheme: light; }' in stylesheet.text
    assert 'window.matchMedia("(prefers-color-scheme: dark)")' in script.text
    assert "window.localStorage.setItem(themeStorageKey, theme)" in script.text
    assert 'themePreference.addEventListener("change"' in script.text
    assert 'themeToggle.setAttribute("aria-pressed", String(dark))' in script.text
    assert 'dark ? "Switch to light theme" : "Switch to dark theme"' in script.text


def test_mathfield_selection_uses_a_light_neutral_palette() -> None:
    """Focused selections should stay readable even when the device prefers dark mode."""
    with TestClient(create_app()) as client:
        stylesheet = client.get("/static/style.css")
    assert "--selection-color: var(--ink)" in stylesheet.text
    assert "--selection-background-color: var(--selection)" in stylesheet.text
    assert "--contains-highlight-color: var(--ink)" in stylesheet.text
    assert "--contains-highlight-background-color: var(--selection-soft)" in stylesheet.text


def test_desktop_solution_uses_the_available_width_with_internal_gutters() -> None:
    """Wide screens should give worked solutions more room without widening the editor."""
    with TestClient(create_app()) as client:
        stylesheet = client.get("/static/style.css")
    assert "@media (min-width: 1100px)" in stylesheet.text
    assert "width: min(1600px, calc(100vw - 64px))" in stylesheet.text
    assert "padding-right: clamp(32px, 3vw, 52px)" in stylesheet.text
    assert "padding-left: clamp(32px, 3vw, 52px)" in stylesheet.text
    assert "border-right: 1px solid var(--line)" in stylesheet.text
    assert "border-left: 1px solid var(--line)" in stylesheet.text


def test_frontend_handles_large_math_and_warms_the_browser_runtime() -> None:
    """Large formulas should scroll, while browser startup happens outside the main thread."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
        runtime = client.get("/static/runtime.mjs")
    assert 'class="math-viewport result-viewport"' in homepage.text
    assert 'aria-label="Scrollable answer"' in homepage.text
    assert 'id="mobile-keyboard-proxy"' in homepage.text
    assert 'id="math-keyboard-button"' in homepage.text
    assert 'inputmode="text"' in homepage.text
    assert "createMathViewport" in script.text
    assert "solverClient.warmup()" in script.text
    assert '"requestIdleCallback" in window' in script.text
    assert "keyboard-sink" not in script.text
    assert 'mobileKeyboardProxy.addEventListener("beforeinput"' in script.text
    assert 'mobileKeyboardProxy.addEventListener("compositionend"' in script.text
    assert 'mobileKeyboardProxy.addEventListener("keydown"' in script.text
    assert 'mobileKeyboardProxy.addEventListener("paste"' in script.text
    assert 'expressionField.executeCommand("deleteBackward")' in script.text
    assert '["^", "^{#0}"]' in script.text
    assert '["_", "_{#0}"]' in script.text
    assert 'selectionMode: template ? "placeholder" : "after"' in script.text
    assert "for (const character of text) insertNativeCharacter(character)" in script.text
    assert "if (usesMobileKeyboard()) mobileKeyboardProxy.blur();" in script.text
    assert "window.mathVirtualKeyboard?.show()" in script.text
    assert "querySelector('[part=\"content\"]')" in script.text
    assert 'classList.add("ML__focused")' in script.text
    assert "window.mathVirtualKeyboard?.hide()" in script.text
    assert "async warmup()" in runtime.text
    assert ".math-viewport" in stylesheet.text
    assert ".primary-math-field::part(virtual-keyboard-toggle)" in stylesheet.text
    assert ".mobile-keyboard-proxy" in stylesheet.text
    assert "font-size: 16px" in stylesheet.text
    assert "max-width: calc(100dvw" not in stylesheet.text
    assert "margin-right: 10px" in stylesheet.text
    assert "contain: inline-size" in stylesheet.text
    assert "-webkit-text-size-adjust: 100%" in stylesheet.text
    assert "@media (hover: none) and (pointer: coarse)" in stylesheet.text
    assert ".primary-math-field { pointer-events: none; }" in stylesheet.text
    assert ".math-keyboard-button { display: inline-flex; }" in stylesheet.text
    assert 'viewBox="0 0 24 24"' in homepage.text
    assert "expressionField.getOffsetFromPoint(event.clientX, event.clientY" in script.text
    assert (
        "expressionField.position = offset >= 0 ? offset : expressionField.lastOffset"
        in script.text
    )
    assert "overscroll-behavior-inline: contain" in stylesheet.text
    assert "touch-action: pan-x pan-y pinch-zoom" in stylesheet.text
    assert "touch-action: pan-x;" not in stylesheet.text
    assert "pointer-events: none" in stylesheet.text
    assert "user-select: none" in stylesheet.text
    assert "-webkit-overflow-scrolling: touch" in stylesheet.text


def test_mobile_sections_use_internal_safe_area_padding() -> None:
    """Mobile sections should provide their own readable, notch-safe gutters."""
    with TestClient(create_app()) as client:
        stylesheet = client.get("/static/style.css")
    assert ".workspace {\n    --mobile-inset-left:" in stylesheet.text
    assert "width: auto" in stylesheet.text
    assert "padding-right: var(--mobile-inset-right)" in stylesheet.text
    assert "padding-left: var(--mobile-inset-left)" in stylesheet.text
    assert "safe-area-inset-right" in stylesheet.text
    assert "safe-area-inset-left" in stylesheet.text


def test_ios_cannot_pan_the_document_horizontally() -> None:
    """The page root should contain overflow while local math panes remain scrollable."""
    with TestClient(create_app()) as client:
        stylesheet = client.get("/static/style.css")
    assert stylesheet.text.count("overflow-x: hidden") >= _MINIMUM_DOCUMENT_OVERFLOW_GUARDS
    assert "overflow-x: clip" not in stylesheet.text
    assert "overscroll-behavior-x: none" in stylesheet.text
    assert "position: relative" in stylesheet.text
    assert ".math-viewport" in stylesheet.text
    assert "overflow-x: auto" in stylesheet.text


def test_mobile_formula_gestures_preserve_vertical_page_scrolling() -> None:
    """Formula panes should reserve native touch panning for the document's vertical axis."""
    with TestClient(create_app()) as client:
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
    assert ".math-viewport { touch-action: pan-y pinch-zoom; }" in stylesheet.text
    assert ".math-viewport > math-field" in stylesheet.text
    assert "max-width: none" in stylesheet.text
    assert "overflow: visible" in stylesheet.text
    assert "overflow-y: visible" in stylesheet.text
    assert "function enableTouchMathScrolling(viewport)" in script.text
    assert "gesture.horizontal = Math.abs(deltaX) > Math.abs(deltaY)" in script.text
    assert "viewport.scrollLeft = gesture.startScrollLeft - deltaX" in script.text
    assert 'enableTouchMathScrolling(document.querySelector(".result-viewport"))' in script.text


def test_mobile_native_keyboard_can_edit_and_extend_systems() -> None:
    """Phone Backspace and Enter should reach MathLive even through the native proxy."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        script = client.get("/static/app.js")
    assert 'enterkeyhint="enter"' in homepage.text
    assert 'const mobileKeyboardSentinel = "\\u2060"' in script.text
    assert "function resetMobileKeyboardProxy()" in script.text
    assert 'expressionField.executeCommand("deleteBackward")' in script.text
    assert 'expressionField.executeCommand("addRowAfter")' in script.text
    assert "if (!addedRow) form.requestSubmit()" in script.text
    assert "if (!mobileKeyboardProxy.value)" in script.text
    assert "now - lastNativeEnterAt < 250" in script.text


def test_desktop_enter_adds_a_row_inside_a_system() -> None:
    """Desktop Enter should extend cases notation instead of submitting the form."""
    with TestClient(create_app()) as client:
        script = client.get("/static/app.js")
    assert "function addSystemRow()" in script.text
    assert 'expressionField.addEventListener("beforeinput"' in script.text
    assert 'event.inputType === "insertLineBreak"' in script.text
    assert 'event.inputType === "insertParagraph"' in script.text
    assert 'const added = expressionField.executeCommand("addRowAfter")' in script.text
    assert "if (addSystemRow()) event.preventDefault()" in script.text


def test_delete_removes_only_an_empty_system_row() -> None:
    """Delete should remove an empty active row without risking populated equations."""
    with TestClient(create_app()) as client:
        script = client.get("/static/app.js")
    assert "function currentMathCellIsEmpty()" in script.text
    assert "expressionField._mathfield?.model" in script.text
    assert "model?.at(model.position)" in script.text
    assert "Array.isArray(atom.parentBranch)" in script.text
    assert "atom.parent?.getCell?.(row, column)" in script.text
    assert 'cellAtom.type === "first"' in script.text
    assert 'cellAtom.type === "placeholder"' in script.text
    assert "function removeEmptySystemRow()" in script.text
    assert 'expressionField.executeCommand("removeRow")' in script.text
    assert 'event.inputType === "deleteContentBackward"' in script.text
    assert 'event.inputType === "deleteContentForward"' in script.text
    assert 'expressionField.executeCommand("deleteBackward")' in script.text
    assert 'expressionField.executeCommand("deleteForward")' in script.text


def test_frontend_renders_domain_constraints_and_accepts_systems() -> None:
    """The graphical client should expose restrictions and system notation."""
    with TestClient(create_app()) as client:
        homepage = client.get("/")
        stylesheet = client.get("/static/style.css")
        script = client.get("/static/app.js")
    assert 'title="System of equations"' in homepage.text
    assert 'data-structure="system"' in homepage.text
    assert r'data-insert="\begin{cases}\placeholder{}\\\placeholder{}\end{cases}"' in homepage.text
    assert 'format: "latex"' in script.text
    assert 'mode: "math"' in script.text
    assert 'key.dataset.structure === "system"' in script.text
    assert "expressionField.setValue(template" in script.text
    assert 'insertionMode: "replaceAll"' in script.text
    assert "createConstraints(step)" in script.text
    assert "Domain restrictions introduced here" in script.text
    assert "step.introduced_constraints" in script.text
    assert ".step-constraints" in stylesheet.text

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\begin{cases}x+y=3\\x-y=1\end{cases}",
                "math_json": [
                    "List",
                    ["Equal", ["Add", "x", "y"], 3],
                    ["Equal", ["Subtract", "x", "y"], 1],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert payload.status == "exact"
    assert tuple(step.rule for step in payload.steps) == (
        "Eliminate x",
        "Solve for y",
        "Substitute back to find x",
    )


def test_web_no_solution_payload_keeps_domain_restrictions() -> None:
    """The web answer should say no solution and retain the excluded value."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\frac{x-1}{x-1}=x",
                "math_json": [
                    "Equal",
                    ["Divide", ["Subtract", "x", 1], ["Subtract", "x", 1]],
                    "x",
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert payload.status == "exact"
    assert payload.result_latex == r"\text{No solution}"
    assert payload.steps[0].introduced_constraints[1].expression_latex == r"x \ne 1"


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


def test_undefined_sum_payload_never_exposes_backend_complex_infinity() -> None:
    """An included singular term should cross the API as a typed undefined result."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"\sum_{n=0}^{\infty}\frac{1}{n^6}",
                "math_json": [
                    "Sum",
                    ["Divide", 1, ["Power", "n", 6]],
                    ["Tuple", "n", 0, "PositiveInfinity"],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert payload.status == "undefined"
    assert payload.result_latex == r"\text{Undefined}"
    assert payload.reason is not None
    assert "n = 0" in payload.reason
    assert "zoo" not in response.text
    assert payload.steps[0].introduced_constraints[1].expression_latex == r"n \ne 0"


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
    assert payload.result_latex == r"x = -2\quad\text{or}\quad x = 2"
    assert [step.rule for step in payload.steps] == [
        "Factor the quadratic",
        "Apply the zero-product property",
        "Solve each factor",
    ]
    assert payload.steps[0].before_latex == "x^{2} - 4 = 0"
    assert payload.steps[-1].after_latex == r"\left[x = 2, x = -2\right]"
    assert all(step.verification_method == "solution-set equivalence" for step in payload.steps)
    assert all(r"\mathtt" not in step.after_latex for step in payload.steps)


def test_rational_cubic_response_uses_a_student_facing_numerical_method() -> None:
    """The browser should lead with rational-root checks and Newton iteration."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/solve",
            json={
                "latex": r"x^2-4=\frac{1}{x+1}",
                "math_json": [
                    "Equal",
                    ["Subtract", ["Power", "x", 2], 4],
                    ["Divide", 1, ["Add", "x", 1]],
                ],
            },
        )
    payload = SolveResponse.model_validate_json(response.text)
    assert payload.status == "exact"
    assert tuple(step.rule for step in payload.steps) == (
        "Multiply both sides by the denominator",
        "Cancel the common factors",
        "Expand and collect like terms",
        "Approximate the real root",
    )
    assert payload.steps[0].introduced_constraints[1].expression_latex == r"x \ne -1"
    assert payload.steps[1].before_latex == (
        r"\left(x + 1\right) \cdot \left(x^{2} - 4\right)"
        r" = \frac{\color{#e93242}{\cancel{x + 1}}}"
        r"{\color{#e93242}{\cancel{x + 1}}}"
    )
    assert tuple(note.label for note in payload.steps[3].notes) == (
        "Rational-root test",
        "Bracket the root",
        "Newton iteration",
        "Successive estimates",
        "Exact form (optional)",
    )
    assert payload.steps[3].notes[1].expression_latex == (
        r"\left[f\left(2\right) = -1, f\left(3\right) = 19\right]"
    )
    assert payload.steps[3].notes[2].expression_latex == (
        r"x_{k+1} = x_k - \frac{f\left(x_k\right)}{f'\left(x_k\right)}"
    )
    assert payload.steps[-1].after_latex == r"x \approx 2.079596"
    assert payload.result_latex == r"x \approx 2.079596"
    assert payload.steps[-1].notes[-1].expression_latex.startswith(r"x = \frac{-1}{3} + \sqrt[3]")


def test_frontend_labels_approximate_results_as_numerical() -> None:
    """An approximate displayed answer should not be labeled as exact."""
    with TestClient(create_app()) as client:
        script = client.get("/static/app.js")
    assert 'payload.result_latex?.includes("\\\\approx")' in script.text
    assert '"Numerical answer"' in script.text


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
