"""Small JSON boundary used by the Pyodide deployment."""

import json
from typing import cast

from stepsolver.errors import QueryError
from stepsolver.mathjson import JsonValue, query_from_mathjson
from stepsolver.presentation import solve_payload
from stepsolver.solver import Solver


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        if all(isinstance(key, str) for key in items):
            return {str(key): _json_value(item) for key, item in items.items()}
    message = "input must contain only JSON values"
    raise QueryError(message)


def solve_mathjson_json(source: str) -> str:
    """Solve one serialized MathJSON query and return a serialized payload."""
    try:
        decoded: object = json.loads(source)
    except json.JSONDecodeError as error:
        message = "input is not valid JSON"
        raise QueryError(message) from error
    result = Solver().solve(query_from_mathjson(_json_value(decoded)))
    return json.dumps(solve_payload(result).as_dict(), ensure_ascii=False)
