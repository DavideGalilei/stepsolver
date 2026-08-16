"""FastAPI application and web-server entry point for StepSolver."""

import os
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from stepsolver.ast import OpaqueExpression
from stepsolver.errors import QueryError
from stepsolver.formatter import format_ascii, format_expression
from stepsolver.latex import format_latex_expression, format_latex_value
from stepsolver.mathjson import query_from_mathjson
from stepsolver.results import ExactResult, SolveResult
from stepsolver.solver import Solver

_ASSET_DIRECTORY = Path(__file__).with_name("web_assets")
_INDEX_FILE = _ASSET_DIRECTORY / "index.html"
_MAX_QUERY_LENGTH = 10_000
_DEFAULT_PORT = 8000
_MAX_PORT = 65_535


class SolveRequest(BaseModel):
    """Validated graphical-editor request for the solve endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latex: str = Field(min_length=1, max_length=_MAX_QUERY_LENGTH)
    math_json: JsonValue


class StepResponse(BaseModel):
    """One browser-renderable verified solution step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int
    rule: str
    explanation: str
    before_ascii: str
    after_ascii: str
    before_latex: str
    after_latex: str
    verification_method: str
    verification_detail: str


class SolveResponse(BaseModel):
    """Typed browser response for an exact or unsolved query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["exact", "unsolved"]
    source: str
    formatted_ascii: str
    result_latex: str | None
    reason: str | None
    steps: tuple[StepResponse, ...]


def _step_responses(result: SolveResult) -> tuple[StepResponse, ...]:
    final_step_number = len(result.steps)
    return tuple(
        StepResponse(
            number=index,
            rule=step.rule,
            explanation=step.explanation,
            before_ascii=format_expression(step.before),
            after_ascii=format_expression(step.after),
            before_latex=format_latex_expression(step.before),
            after_latex=(
                format_latex_value(result.value)
                if (
                    isinstance(result, ExactResult)
                    and index == final_step_number
                    and isinstance(step.after, OpaqueExpression)
                )
                else format_latex_expression(step.after)
            ),
            verification_method=step.verification.method.value,
            verification_detail=step.verification.detail,
        )
        for index, step in enumerate(result.steps, start=1)
    )


def _solve_response(result: SolveResult) -> SolveResponse:
    steps = _step_responses(result)
    if isinstance(result, ExactResult):
        return SolveResponse(
            status="exact",
            source=result.query.source,
            formatted_ascii=format_ascii(result),
            result_latex=format_latex_value(result.value),
            reason=None,
            steps=steps,
        )
    return SolveResponse(
        status="unsolved",
        source=result.query.source,
        formatted_ascii=format_ascii(result),
        result_latex=None,
        reason=result.reason,
        steps=steps,
    )


def homepage() -> HTMLResponse:
    """Serve the StepSolver single-page frontend."""
    return HTMLResponse(_INDEX_FILE.read_text(encoding="utf-8"))


def solve_endpoint(payload: SolveRequest) -> SolveResponse:
    """Parse and solve one graphical-editor query for the browser."""
    try:
        result = Solver().solve(query_from_mathjson(payload.math_json))
    except QueryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _solve_response(result)


def create_app() -> FastAPI:
    """Construct the web application without global mutable solver state."""
    application = FastAPI(
        title="StepSolver",
        description="Strictly typed symbolic mathematics with verified steps.",
        version="0.1.0",
    )
    application.mount("/static", StaticFiles(directory=_ASSET_DIRECTORY), name="static")
    application.add_api_route("/", homepage, methods=["GET"], response_class=HTMLResponse)
    application.add_api_route(
        "/api/solve",
        solve_endpoint,
        methods=["POST"],
        response_model=SolveResponse,
    )
    return application


app = create_app()


def serve(*, host: str, port: int, reload: bool) -> None:
    """Start Uvicorn for the exported StepSolver ASGI application."""
    uvicorn.run("stepsolver.web:app", host=host, port=port, reload=reload)


def main() -> None:
    """Run the development web server using environment-based settings."""
    host = os.environ.get("STEPSOLVER_HOST", "127.0.0.1")
    port_text = os.environ.get("STEPSOLVER_PORT", str(_DEFAULT_PORT))
    if not port_text.isdecimal():
        message = "STEPSOLVER_PORT must be a positive integer"
        raise ValueError(message)
    port = int(port_text)
    if not 1 <= port <= _MAX_PORT:
        message = "STEPSOLVER_PORT must be between 1 and 65535"
        raise ValueError(message)
    reload_enabled = os.environ.get("STEPSOLVER_RELOAD", "false").lower() in {"1", "true", "yes"}
    serve(host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
