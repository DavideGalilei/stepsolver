"""Backend-native display objects shared by derivation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from stepsolver.results import VerificationMethod

type EquationBackendExpression = sp.Basic | tuple[sp.Basic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIntegral:
    """An unevaluated backend integral used only for derivation display."""

    integrand: sp.Basic
    variable: sp.Symbol
    coefficient: sp.Basic | None = None
    lower: sp.Basic | None = None
    upper: sp.Basic | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDifferential:
    """A displayed differential such as dx or du."""

    variable: sp.Symbol
    coefficient: sp.Basic | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDerivative:
    """A displayed derivative of a backend expression."""

    expression: sp.Basic
    variable: sp.Symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIntegrationByPartsRule:
    """The generic integration-by-parts identity."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendQuadraticSolutions:
    """Two roots displayed as quadratic-formula fractions."""

    variable: sp.Symbol
    negative_numerator: BackendExpression
    positive_numerator: BackendExpression
    denominator: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendCardanoSolution:
    """The real Cardano root of a cubic with positive discriminant."""

    variable: sp.Symbol
    shift: sp.Basic
    first_radicand: sp.Basic
    second_radicand: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendNewtonRule:
    """The generic Newton iteration identity."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendNewtonIterations:
    """Successive Newton approximations with indexed variables."""

    variable: sp.Symbol
    values: tuple[sp.Basic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendApproximateSolutions:
    """One or more numerical roots displayed with approximation signs."""

    variable: sp.Symbol
    roots: tuple[sp.Basic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendCrossedOut:
    """A factor displayed with cancellation strokes."""

    expression: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIntroducedProduct:
    """A newly introduced multiplication wrapped around an existing expression."""

    multiplier: BackendExpression
    expression: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendEvaluationAtBounds:
    """An antiderivative evaluated between lower and upper bounds."""

    expression: sp.Basic
    variable: sp.Symbol
    lower: sp.Basic
    upper: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendEvaluationAtIndex:
    """A sequence term evaluated at one index."""

    expression: BackendExpression
    variable: sp.Symbol
    index: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendSigma:
    """A displayed finite or infinite summation."""

    expression: BackendExpression
    variable: sp.Symbol
    lower: sp.Basic
    upper: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendUndefined:
    """A displayed undefined mathematical value."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendLimit:
    """A displayed one- or two-sided limit."""

    expression: BackendExpression
    variable: sp.Symbol
    point: sp.Basic
    direction: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendNotEqual:
    """A displayed non-equality between two backend expressions."""

    left: BackendExpression
    right: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendSum:
    """A displayed sum containing backend and derivation expressions."""

    terms: tuple[BackendExpression, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendProduct:
    """A displayed product containing backend and derivation expressions."""

    factors: tuple[BackendExpression, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendQuotient:
    """A displayed quotient containing backend and derivation expressions."""

    numerator: BackendExpression
    denominator: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDifference:
    """A displayed subtraction containing backend and derivation expressions."""

    left: BackendExpression
    right: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendIdentity:
    """A displayed equality between two backend derivation expressions."""

    left: BackendExpression
    right: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendSystem:
    """A group of equations displayed with a single system brace."""

    equations: tuple[sp.Basic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendRowOperation:
    """One elementary row replacement used during elimination."""

    target: int
    source: int
    factor: sp.Basic


type BackendExpression = (
    EquationBackendExpression
    | BackendIntegral
    | BackendDifferential
    | BackendDerivative
    | BackendIntegrationByPartsRule
    | BackendQuadraticSolutions
    | BackendCardanoSolution
    | BackendNewtonRule
    | BackendNewtonIterations
    | BackendApproximateSolutions
    | BackendCrossedOut
    | BackendIntroducedProduct
    | BackendEvaluationAtBounds
    | BackendEvaluationAtIndex
    | BackendSigma
    | BackendUndefined
    | BackendLimit
    | BackendNotEqual
    | BackendSystem
    | BackendRowOperation
    | BackendSum
    | BackendProduct
    | BackendQuotient
    | BackendDifference
    | BackendIdentity
)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendMathNote:
    """A labeled mathematical annotation supporting a derivation step."""

    label: str
    expression: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendStepConstraint:
    """A domain condition introduced by a backend derivation step."""

    explanation: str
    expression: BackendExpression


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendDerivationStep:
    """One backend-native transformation awaiting conversion to the public AST."""

    rule: str
    before: BackendExpression
    after: BackendExpression
    explanation: str
    verification_method: VerificationMethod
    verification_detail: str
    notes: tuple[BackendMathNote, ...] = ()
    introduced_constraints: tuple[BackendStepConstraint, ...] = ()
