"""Command-line interface for StepSolver."""

import argparse
import sys
from collections.abc import Sequence

from stepsolver.errors import StepSolverError
from stepsolver.formatter import format_ascii
from stepsolver.results import UnsolvedResult
from stepsolver.solver import Solver


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stepsolver",
        description="Solve an ASCII mathematical query with verified steps.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="query such as 'integrate(sin(x), x, 0, pi)'; stdin is used when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the StepSolver command-line interface."""
    arguments = _build_parser().parse_args(argv)
    source = arguments.query
    if source is None:
        source = sys.stdin.read().strip()
    try:
        result = Solver().solve(source)
    except StepSolverError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(format_ascii(result))
    return 1 if isinstance(result, UnsolvedResult) else 0


if __name__ == "__main__":
    raise SystemExit(main())
