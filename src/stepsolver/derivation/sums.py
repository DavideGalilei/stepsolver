"""Human-readable derivations and domain checks for summations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendEvaluationAtIndex,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendQuotient,
    BackendSigma,
    BackendStepConstraint,
    BackendUndefined,
)
from stepsolver.derivation.sums_convergence import CONVERGENCE_SUM_STRATEGIES
from stepsolver.derivation.sums_factorial import FACTORIAL_SUM_STRATEGIES
from stepsolver.derivation.sums_series import CLOSED_FORM_SUM_STRATEGIES
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import is_object_sequence

_SQUARE_POWER = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class UndefinedSummation:
    """One included index at which a summand is undefined."""

    index: sp.Basic
    denominator: sp.Basic
    steps: tuple[BackendDerivationStep, ...]


def _included_integer(
    value: sp.Basic,
    lower: sp.Basic,
    upper: sp.Basic,
) -> bool:
    if value.is_integer is not True:
        return False
    above_lower = sp.simplify(value - lower).is_nonnegative is True
    below_upper = upper == sp.oo or sp.simplify(upper - value).is_nonnegative is True
    return above_lower and below_upper


def find_undefined_summation(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
) -> UndefinedSummation | None:
    """Find the first denominator zero included in a discrete summation range."""
    _raw_numerator, raw_denominator = sp.fraction(expression)
    if raw_denominator == sp.Integer(0) and _included_integer(lower, lower, upper):
        sigma = BackendSigma(
            expression=BackendQuotient(
                numerator=sp.Integer(1),
                denominator=sp.Integer(0),
            ),
            variable=variable,
            lower=lower,
            upper=upper,
        )
        return UndefinedSummation(
            index=lower,
            denominator=raw_denominator,
            steps=(
                BackendDerivationStep(
                    rule="Check the summand's domain",
                    before=sigma,
                    after=BackendUndefined(),
                    explanation="The summand has a zero denominator at every index.",
                    verification_method=VerificationMethod.EXACT_ARITHMETIC,
                    verification_detail="Division by zero is undefined.",
                    introduced_constraints=(
                        BackendStepConstraint(
                            explanation="The summand's denominator must be nonzero.",
                            expression=BackendNotEqual(
                                left=raw_denominator,
                                right=sp.Integer(0),
                            ),
                        ),
                    ),
                ),
            ),
        )
    _numerator, denominator = sp.fraction(sp.together(expression))
    if denominator == sp.Integer(1):
        return None
    try:
        roots = sp.solve(denominator, variable)
    except (NotImplementedError, TypeError, ValueError):
        return None
    if not is_object_sequence(roots):
        return None
    included = tuple(
        root
        for root in roots
        if isinstance(root, sp.Basic) and _included_integer(root, lower, upper)
    )
    if not included:
        return None
    index = included[0]
    sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    undefined_term = BackendIdentity(
        left=BackendEvaluationAtIndex(
            expression=expression,
            variable=variable,
            index=index,
        ),
        right=BackendUndefined(),
    )
    constraints = (
        BackendStepConstraint(
            explanation="The summand's denominator must be nonzero.",
            expression=BackendNotEqual(left=denominator, right=sp.Integer(0)),
        ),
        BackendStepConstraint(
            explanation="This index is excluded from the summand's domain.",
            expression=BackendNotEqual(left=variable, right=index),
        ),
    )
    return UndefinedSummation(
        index=index,
        denominator=denominator,
        steps=(
            BackendDerivationStep(
                rule="Check the summand's domain",
                before=sigma,
                after=undefined_term,
                explanation=(
                    f"The range includes {variable} = {index}, where the summand has a zero "
                    "denominator."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="Direct substitution produces an undefined term.",
                introduced_constraints=constraints,
            ),
        ),
    )


def _power_sum_formula(power: int, upper: sp.Basic) -> sp.Basic:
    if power == 1:
        return upper * (upper + 1) / 2
    if power == _SQUARE_POWER:
        return upper * (upper + 1) * (2 * upper + 1) / 6
    return (upper * (upper + 1) / 2) ** 2


def _display_power_sum_formula(power: int, upper: sp.Basic) -> BackendProduct | BackendQuotient:
    consecutive = BackendProduct(factors=(upper, upper + 1))
    half = BackendQuotient(numerator=consecutive, denominator=sp.Integer(2))
    if power == 1:
        return half
    if power == _SQUARE_POWER:
        return BackendQuotient(
            numerator=BackendProduct(factors=(upper, upper + 1, 2 * upper + 1)),
            denominator=sp.Integer(6),
        )
    return BackendProduct(factors=(half, half))


def _derive_power_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if lower not in {sp.Integer(0), sp.Integer(1)} or upper.is_integer is not True:
        return ()
    power = next(
        (
            candidate
            for candidate in (1, 2, 3)
            if sp.simplify(expression - variable**candidate) == sp.Integer(0)
        ),
        None,
    )
    if power is None:
        return ()
    generic_upper = sp.Symbol("N", integer=True, positive=True)
    formula = _display_power_sum_formula(power, upper)
    generic_identity = BackendIdentity(
        left=BackendSigma(
            expression=variable**power,
            variable=variable,
            lower=sp.Integer(1),
            upper=generic_upper,
        ),
        right=_power_sum_formula(power, generic_upper),
    )
    labels = {1: "first integers", 2: "squares", 3: "cubes"}
    return (
        BackendDerivationStep(
            rule=f"Use the sum of {labels[power]} identity",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=formula,
            explanation=f"Substitute N = {upper} into the standard power-sum identity.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The finite power-sum identity was applied exactly.",
            notes=(BackendMathNote(label="Identity", expression=generic_identity),),
        ),
        BackendDerivationStep(
            rule="Simplify the arithmetic",
            before=formula,
            after=result,
            explanation="Evaluate the products and division.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The displayed arithmetic equals the exact result.",
        ),
    )


def derive_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive common finite sums and convergent or divergent series."""
    for strategy in (
        _derive_power_sum,
        *FACTORIAL_SUM_STRATEGIES,
        *CLOSED_FORM_SUM_STRATEGIES,
        *CONVERGENCE_SUM_STRATEGIES,
    ):
        derivation = strategy(expression, variable, lower, upper, result)
        if derivation:
            return derivation
    return ()
