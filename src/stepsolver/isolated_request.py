"""Strict JSON decoding for the isolated solver process."""

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from stepsolver.errors import ParseError


class _SolveRequestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expression: str

    @field_validator("expression")
    @classmethod
    def require_ascii_expression(cls, value: str) -> str:
        if not value.isascii():
            message = "expression must contain only ASCII characters"
            raise ValueError(message)
        return value


def parse_request_expression(document: bytes) -> str:
    """Extract the ASCII expression from a complete worker request."""
    try:
        return _SolveRequestPayload.model_validate_json(document).expression
    except ValidationError as error:
        message = "request must contain one ASCII expression"
        raise ParseError(message, position=0) from error
