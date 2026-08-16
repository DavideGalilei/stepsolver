"""FastAPI application and web-server entry point for StepSolver."""

import os
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from stepsolver.errors import QueryError
from stepsolver.mathjson import JsonValue, query_from_mathjson
from stepsolver.presentation import solve_payload
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


class StepNoteResponse(BaseModel):
    """One labeled mathematical rule or substitution supporting a step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    expression_ascii: str
    expression_latex: str


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
    notes: tuple[StepNoteResponse, ...]


class SolveResponse(BaseModel):
    """Typed browser response for an exact, divergent, or unsolved query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["exact", "divergent", "unsolved"]
    source: str
    formatted_ascii: str
    result_latex: str | None
    reason: str | None
    steps: tuple[StepResponse, ...]


def homepage() -> HTMLResponse:
    """Serve the StepSolver single-page frontend."""
    return HTMLResponse(_INDEX_FILE.read_text(encoding="utf-8"))


def solve_endpoint(payload: SolveRequest) -> SolveResponse:
    """Parse and solve one graphical-editor query for the browser."""
    try:
        result = Solver().solve(query_from_mathjson(payload.math_json))
    except QueryError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return SolveResponse.model_validate(solve_payload(result).as_dict())


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
