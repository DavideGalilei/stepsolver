"""One-request stdin/stdout adapter for disposable StepSolver workers."""

import json
import sys
from collections.abc import Sequence

from stepsolver.errors import BackendError, LimitCategory, LimitError, ParseError, QueryError
from stepsolver.isolated import (
    DEFAULT_ISOLATED_LIMITS,
    IsolatedErrorCode,
    IsolatedFailure,
    IsolatedLimits,
    IsolatedSuccess,
    parse_isolated_request,
    solve_isolated,
)


def _encoded(value: IsolatedSuccess | IsolatedFailure) -> bytes:
    return json.dumps(value.as_dict(), ensure_ascii=True, separators=(",", ":")).encode()


def _failure(code: IsolatedErrorCode, message: str) -> bytes:
    return _encoded(IsolatedFailure(ok=False, code=code, message=message))


def _limit_code(error: LimitError) -> IsolatedErrorCode:
    match error.violation.category:
        case LimitCategory.INPUT:
            return IsolatedErrorCode.INPUT_LIMIT
        case LimitCategory.EXPRESSION:
            return IsolatedErrorCode.EXPRESSION_LIMIT
        case LimitCategory.RESULT:
            return IsolatedErrorCode.RESULT_LIMIT


def run(
    document: bytes,
    limits: IsolatedLimits = DEFAULT_ISOLATED_LIMITS,
) -> bytes:
    """Handle one complete request document without reading global process state."""
    try:
        request = parse_isolated_request(document, limits)
        solution = solve_isolated(request, limits=limits)
        return _encoded(IsolatedSuccess(ok=True, solution=solution))
    except LimitError as error:
        return _failure(_limit_code(error), str(error))
    except (ParseError, QueryError):
        return _failure(IsolatedErrorCode.INVALID_REQUEST, "The expression is invalid.")
    except BackendError:
        return _failure(
            IsolatedErrorCode.SOLVER_FAILED, "The solver could not complete the request."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Read one bounded request, write one response, and exit."""
    if argv:
        return 2
    maximum = DEFAULT_ISOLATED_LIMITS.request_bytes
    document = sys.stdin.buffer.read(maximum + 1)
    sys.stdout.buffer.write(run(document))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
