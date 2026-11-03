"""Entry point that explains how to install optional web dependencies."""

from __future__ import annotations

import importlib

_WEB_DEPENDENCIES = frozenset({"fastapi", "stepsolver.web", "uvicorn"})


def main() -> None:
    """Start the web server, or explain which optional extra is missing."""
    try:
        web_module = importlib.import_module("stepsolver.web")
    except ModuleNotFoundError as error:
        if error.name not in _WEB_DEPENDENCIES:
            raise
        message = (
            "The StepSolver web interface requires optional dependencies. "
            "Install them with: pip install 'stepsolver[web]'"
        )
        raise SystemExit(message) from None
    web_module.main()


if __name__ == "__main__":
    main()
