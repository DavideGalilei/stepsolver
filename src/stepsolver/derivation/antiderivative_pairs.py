"""Elementary function and antiderivative pairs used by multiple rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class AntiderivativePair:
    """A function and its elementary primitive, with classroom rule labels."""

    function: Callable[[sp.Basic], sp.Basic]
    primitive: Callable[[sp.Basic], sp.Basic]
    name: str
    reverse_chain_rule: str


def _negative_cosine(argument: sp.Basic) -> sp.Basic:
    return -sp.cos(argument)


ELEMENTARY_PAIRS = (
    AntiderivativePair(sp.sin, _negative_cosine, "sine", "Apply the reverse chain rule for sine"),
    AntiderivativePair(sp.cos, sp.sin, "cosine", "Apply the reverse chain rule for cosine"),
    AntiderivativePair(
        sp.exp, sp.exp, "exponential", "Apply the reverse chain rule for the exponential"
    ),
)
