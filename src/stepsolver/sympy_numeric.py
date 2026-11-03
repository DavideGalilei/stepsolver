"""Bounded real numerical root finding with human-readable Newton steps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendApproximateSolutions,
    BackendDerivationStep,
    BackendDerivative,
    BackendExpression,
    BackendIdentity,
    BackendMathNote,
    BackendNewtonIterations,
    BackendNewtonRule,
)
from stepsolver.errors import BackendError
from stepsolver.results import VerificationMethod

_DEFAULT_TOLERANCE = 1e-12
_MAX_ITERATIONS = 50
_DISPLAY_DIGITS = 15
_WORK_DIGITS = 30
_MAX_DISPLAY_ITERATIONS = 4
_MAX_POLYNOMIAL_DEGREE = 20
_MINIMUM_TRACE_LENGTH = 2


class NumericSolveError(BackendError):
    """A numerical solve failure with a user-facing explanation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class NewtonTrace:
    """A converged Newton iteration trace and its final residual."""

    values: tuple[sp.Basic, ...]
    residual: sp.Basic


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericRoot:
    """One verified real root and the iteration trace that found it."""

    value: sp.Basic
    display: str
    residual: float
    trace: tuple[sp.Basic, ...]
    initial_guess: sp.Basic | None


@dataclass(frozen=True, slots=True, kw_only=True)
class NumericSolve:
    """The result of a bounded real numerical solve."""

    roots: tuple[NumericRoot, ...]
    steps: tuple[BackendDerivationStep, ...]
    method: str
    tolerance: float


def solve_numeric(
    *,
    original: sp.Basic,
    residual: sp.Basic,
    variable: sp.Symbol,
    guess: sp.Basic | None,
    from_equation: bool,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = _MAX_ITERATIONS,
) -> NumericSolve:
    """Solve one real numerical root problem with bounded Newton iterations."""
    if residual == sp.Integer(0):
        raise NumericSolveError(
            "The residual is identically zero, so every real value satisfies the equation. "
            "A numerical root is not needed."
        )
    if not residual.free_symbols:
        raise NumericSolveError(
            "The residual does not depend on the selected variable, so it has no numerical "
            "root to find."
        )
    if not residual.free_symbols <= {variable}:
        raise NumericSolveError(
            "Numerical solving currently supports one real variable with numeric coefficients."
        )

    derivative = sp.diff(residual, variable)
    steps: list[BackendDerivationStep] = []
    residual_step = _residual_step(original, residual, variable, from_equation=from_equation)
    if residual_step is not None:
        steps.append(residual_step)

    if guess is not None:
        initial = _real_float(guess, role="initial guess")
        trace = _newton_trace(
            residual,
            derivative,
            variable,
            initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        root = NumericRoot(
            value=trace.values[-1],
            display=_format_float(trace.values[-1]),
            residual=_abs_float(trace.residual),
            trace=trace.values,
            initial_guess=initial,
        )
        steps.extend(_newton_steps(variable, root, tolerance=tolerance))
        return NumericSolve(
            roots=(root,),
            steps=tuple(steps),
            method="bounded Newton iteration",
            tolerance=tolerance,
        )

    polynomial_match = _polynomial_real_roots(residual, variable)
    if polynomial_match is None:
        raise NumericSolveError(
            "An initial guess is required for a general transcendental equation. "
            "Provide guess=... to choose the starting point, or use solve for an exact "
            "symbolic result when one is available."
        )
    polynomial, real_roots = polynomial_match
    steps.append(_isolation_step(residual, variable, polynomial, real_roots))
    if not real_roots:
        return NumericSolve(
            roots=(),
            steps=tuple(steps),
            method="polynomial real-root isolation",
            tolerance=tolerance,
        )

    numeric_roots: list[NumericRoot] = []
    first_trace: NumericRoot | None = None
    for root_value in real_roots:
        refinement, candidate = _polynomial_trace(
            residual,
            derivative,
            variable,
            root_value,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        residual_value: float
        if refinement is not None:
            refined_value = refinement.values[-1]
            residual_value = _abs_float(refinement.residual)
            numeric_root = NumericRoot(
                value=refined_value,
                display=_format_float(refined_value),
                residual=residual_value,
                trace=refinement.values,
                initial_guess=candidate,
            )
        else:
            residual_value = _abs_float(_evaluate(residual, variable, root_value))
            numeric_root = NumericRoot(
                value=root_value,
                display=_format_float(root_value),
                residual=residual_value,
                trace=(),
                initial_guess=None,
            )
        if residual_value > tolerance:
            raise NumericSolveError(
                "A real root was isolated, but its residual did not meet the numerical tolerance."
            )
        numeric_roots.append(numeric_root)
        if first_trace is None:
            first_trace = numeric_root

    if first_trace is not None:
        steps.extend(_newton_steps(variable, first_trace, tolerance=tolerance))
    method = (
        "polynomial real-root isolation with Newton refinement"
        if any(root.trace for root in numeric_roots)
        else "polynomial real-root isolation"
    )
    return NumericSolve(
        roots=tuple(numeric_roots),
        steps=tuple(steps),
        method=method,
        tolerance=tolerance,
    )


def _residual_step(
    original: sp.Basic,
    residual: sp.Basic,
    variable: sp.Symbol,
    *,
    from_equation: bool,
) -> BackendDerivationStep | None:
    if (
        isinstance(original, sp.Equality)
        and getattr(original.rhs, "is_zero", None) is True
        and sp.simplify(original.lhs - residual) == sp.Integer(0)
    ):
        return None
    before: BackendExpression = (
        BackendIdentity(left=original.lhs, right=original.rhs)
        if isinstance(original, sp.Equality)
        else original
    )
    after = BackendIdentity(left=residual, right=sp.Integer(0))
    function = sp.Function("f")
    explanation = (
        "Move all terms to one side so the numerical method solves f(x) = 0."
        if from_equation
        else "A root of the expression is a value where the residual equals zero."
    )
    detail = (
        "Moving all terms to one side preserves the equation's real solutions."
        if from_equation
        else "Setting the residual equal to zero is the definition of a root."
    )
    return BackendDerivationStep(
        rule="Form the residual equation",
        before=before,
        after=after,
        explanation=explanation,
        verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
        verification_detail=detail,
        notes=(
            BackendMathNote(
                label="Residual",
                expression=BackendIdentity(left=function(variable), right=residual),
            ),
            BackendMathNote(
                label="Derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=function(variable), variable=variable),
                    right=sp.diff(residual, variable),
                ),
            ),
        ),
    )


def _isolation_step(
    residual: sp.Basic,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
) -> BackendDerivationStep:
    degree = polynomial.degree()
    before = BackendIdentity(left=residual, right=sp.Integer(0))
    notes = (
        BackendMathNote(label="Polynomial degree", expression=sp.Integer(degree)),
        BackendMathNote(label="Real root count", expression=sp.Integer(len(roots))),
    )
    if not roots:
        return BackendDerivationStep(
            rule="Check for real roots",
            before=before,
            after=(),
            explanation=(
                f"The residual is a polynomial of degree {degree}. Real-root isolation "
                "finds no real roots; complex roots are not returned because numerical "
                "solving is restricted to the real domain."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="SymPy real-root isolation found no real roots.",
            notes=notes,
        )
    return BackendDerivationStep(
        rule="Isolate the real roots",
        before=before,
        after=BackendApproximateSolutions(
            variable=variable,
            roots=roots,
            digits=_DISPLAY_DIGITS,
        ),
        explanation=(
            f"The residual is a polynomial of degree {degree}. Real-root isolation finds "
            f"{len(roots)} real root(s); complex roots are excluded because numerical "
            "solving is restricted to the real domain."
        ),
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail=(
            "SymPy real-root isolation returned only real roots; each displayed "
            "approximation is checked by substitution."
        ),
        notes=notes,
    )


def _newton_steps(
    variable: sp.Symbol,
    root: NumericRoot,
    *,
    tolerance: float,
) -> tuple[BackendDerivationStep, ...]:
    if not root.trace:
        return ()
    if len(root.trace) < _MINIMUM_TRACE_LENGTH:
        return (
            BackendDerivationStep(
                rule="Check the initial guess",
                before=BackendNewtonIterations(
                    variable=variable,
                    values=(root.trace[0],),
                    digits=_DISPLAY_DIGITS,
                ),
                after=BackendApproximateSolutions(
                    variable=variable,
                    roots=(root.value,),
                    digits=_DISPLAY_DIGITS,
                ),
                explanation=(
                    "The initial guess already satisfies the residual tolerance, so no "
                    "Newton update is needed."
                ),
                verification_method=VerificationMethod.SUBSTITUTION,
                verification_detail=(
                    "Substituting the initial guess into the residual gives "
                    f"|f(x)| = {_format_residual(root.residual)}, within the "
                    f"{tolerance:g} tolerance."
                ),
            ),
        )
    displayed = root.trace[:_MAX_DISPLAY_ITERATIONS]
    notes: tuple[BackendMathNote, ...] = (
        BackendMathNote(label="Newton update", expression=BackendNewtonRule()),
    )
    apply_step = BackendDerivationStep(
        rule="Apply Newton's method",
        before=BackendNewtonIterations(
            variable=variable,
            values=(root.trace[0],),
            digits=_DISPLAY_DIGITS,
        ),
        after=BackendNewtonIterations(
            variable=variable,
            values=displayed,
            digits=_DISPLAY_DIGITS,
        ),
        explanation=(
            "Use Newton's update with the initial guess shown. Each iterate is computed "
            "from the previous one; continue until the residual is within tolerance."
        ),
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail=(
            "Each displayed iterate was computed from the Newton update at 30-digit "
            "working precision."
        ),
        notes=notes,
    )
    check_step = BackendDerivationStep(
        rule="Check the numerical residual",
        before=BackendNewtonIterations(
            variable=variable,
            values=displayed,
            digits=_DISPLAY_DIGITS,
        ),
        after=BackendApproximateSolutions(
            variable=variable,
            roots=(root.value,),
            digits=_DISPLAY_DIGITS,
        ),
        explanation=(
            f"Continuing the same iteration until the residual is within the {tolerance:g} "
            f"tolerance gives x approximately {root.display}."
        ),
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail=(
            "Substituting the final approximation into the residual gives "
            f"|f(x)| = {_format_residual(root.residual)}, within the {tolerance:g} tolerance."
        ),
    )
    return (apply_step, check_step)


def _newton_trace(
    residual: sp.Basic,
    derivative: sp.Basic,
    variable: sp.Symbol,
    initial: sp.Basic,
    *,
    tolerance: float,
    max_iterations: int,
) -> NewtonTrace:
    current = sp.Float(str(sp.N(initial, _WORK_DIGITS)), _WORK_DIGITS)
    values: list[sp.Basic] = [current]
    residual_value = _evaluate(residual, variable, current)
    for _index in range(max_iterations):
        if _abs_float(residual_value) <= tolerance:
            return NewtonTrace(values=tuple(values), residual=residual_value)
        slope = _evaluate(derivative, variable, current)
        if _abs_float(slope) <= 1e-30:
            raise NumericSolveError(
                "Newton's method stopped because the derivative is zero at "
                f"x = {_format_float(current)}."
            )
        next_value = sp.N(current - residual_value / slope, _WORK_DIGITS)
        if not _is_finite_real(next_value):
            raise NumericSolveError(
                "Newton's method left the real domain or produced a non-finite value."
            )
        previous = current
        current = sp.Float(str(sp.N(next_value, _WORK_DIGITS)), _WORK_DIGITS)
        values.append(current)
        residual_value = _evaluate(residual, variable, current)
        if _distance(current, previous) <= tolerance and _abs_float(residual_value) <= tolerance:
            return NewtonTrace(values=tuple(values), residual=residual_value)
    raise NumericSolveError(f"Newton's method did not converge within {max_iterations} iterations.")


def _polynomial_real_roots(
    residual: sp.Basic,
    variable: sp.Symbol,
) -> tuple[sp.Poly, tuple[sp.Basic, ...]] | None:
    try:
        polynomial = sp.Poly(residual, variable)
    except sp.PolynomialError:
        return None
    degree = polynomial.degree()
    if degree < 1:
        return None
    if degree > _MAX_POLYNOMIAL_DEGREE:
        raise NumericSolveError(
            f"The polynomial degree {degree} is too high for the bounded numerical path. "
            "Provide an initial guess or use the symbolic solver."
        )
    try:
        roots = sp.real_roots(residual)
    except (NotImplementedError, sp.PolynomialError) as error:
        raise NumericSolveError(
            "Real-root isolation failed for this polynomial. Provide an initial guess."
        ) from error
    unique: list[sp.Basic] = []
    for root in roots:
        numeric = sp.N(root, _WORK_DIGITS)
        if not _is_finite_real(numeric):
            continue
        if any(_distance(numeric, existing) <= 1e-18 for existing in unique):
            continue
        unique.append(numeric)
    unique.sort(key=lambda value: float(str(sp.N(value, _WORK_DIGITS))))
    return polynomial, tuple(unique)


def _polynomial_trace(
    residual: sp.Basic,
    derivative: sp.Basic,
    variable: sp.Symbol,
    root: sp.Basic,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[NewtonTrace | None, sp.Basic | None]:
    for candidate in _candidate_guesses(root):
        try:
            trace = _newton_trace(
                residual,
                derivative,
                variable,
                candidate,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        except NumericSolveError:
            continue
        if _distance(trace.values[-1], root) <= 1e-7:
            return trace, candidate
    return None, None


def _candidate_guesses(root: sp.Basic) -> tuple[sp.Basic, ...]:
    base = math.floor(float(str(sp.N(root, _WORK_DIGITS))))
    raw = (
        sp.Integer(base),
        sp.Integer(base + 1),
        sp.Float(base) + sp.Rational(1, 2),
        sp.Integer(base - 1),
        sp.Integer(base + 2),
    )
    unique: list[sp.Basic] = []
    for candidate in raw:
        numeric = sp.N(candidate, _WORK_DIGITS)
        if _distance(numeric, root) <= 1e-6:
            continue
        if any(_distance(numeric, existing) <= 1e-20 for existing in unique):
            continue
        unique.append(numeric)
    return tuple(unique)


def _evaluate(expression: sp.Basic, variable: sp.Symbol, value: sp.Basic) -> sp.Basic:
    try:
        result = sp.N(expression.subs(variable, value), _WORK_DIGITS + 5)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise NumericSolveError(
            "The residual could not be evaluated at a numerical iterate."
        ) from error
    if not _is_finite_real(result):
        raise NumericSolveError("The residual is undefined or non-real at a numerical iterate.")
    return result


def _real_float(value: sp.Basic, *, role: str) -> sp.Basic:
    try:
        numeric = sp.N(value, _WORK_DIGITS)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise NumericSolveError(f"The {role} must be a finite real number.") from error
    if not _is_finite_real(numeric):
        raise NumericSolveError(f"The {role} must be a finite real number.")
    return sp.Float(str(sp.N(numeric, _WORK_DIGITS)), _WORK_DIGITS)


def _is_finite_real(value: sp.Basic) -> bool:
    if value in {sp.nan, sp.zoo, sp.oo, -sp.oo}:
        return False
    if value.is_real is not True:
        return False
    return getattr(value, "is_finite", None) is not False


def _abs_float(value: sp.Basic) -> float:
    return abs(float(str(sp.N(value, _WORK_DIGITS))))


def _distance(left: sp.Basic, right: sp.Basic) -> float:
    return _abs_float(left - right)


def _format_float(value: sp.Basic, digits: int = _DISPLAY_DIGITS) -> str:
    text = str(sp.N(value, digits))
    if "e" not in text.lower() and "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _format_residual(value: float) -> str:
    return format(abs(value), ".3g")
