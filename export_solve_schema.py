"""Export the JSON Schema used to compile the browser response validator."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from stepsolver.browser_result import SolvePayload


def main() -> None:
    """Write the serialized Python record shape for the Node build."""
    schema = TypeAdapter(SolvePayload).json_schema(mode="serialization")
    target = Path(__file__).with_name("frontend") / "solve-result.schema.json"
    target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
