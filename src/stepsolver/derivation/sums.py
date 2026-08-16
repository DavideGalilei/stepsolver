"""Human-readable derivations and domain checks for summations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendEvaluationAtIndex,
    BackendExpression,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendQuotient,
    BackendSigma,
    BackendStepConstraint,
    BackendUndefined,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_support import is_object_sequence

_SQUARE_POWER = 2
_GEOMETRIC_POWER_ARITY = 2


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
    roots = sp.solve(denominator, variable)
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


def _p_series_power(expression: sp.Basic, variable: sp.Symbol) -> int | None:
    return next(
        (
            power
            for power in range(1, 13)
            if sp.simplify(expression - variable ** (-power)) == sp.Integer(0)
        ),
        None,
    )


def _derive_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if lower != sp.Integer(1) or upper != sp.oo:
        return ()
    power = _p_series_power(expression, variable)
    if power is None:
        return ()
    generic_power = sp.Symbol("p", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=variable ** (-generic_power),
            variable=variable,
            lower=sp.Integer(1),
            upper=sp.oo,
        ),
        right=sp.zeta(generic_power),
    )
    convergence = sp.Gt(generic_power, sp.Integer(1), evaluate=False)
    sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if power <= 1:
        return (
            BackendDerivationStep(
                rule="Apply the p-series test",
                before=sigma,
                after=result,
                explanation="A p-series converges only when p > 1; here p = 1.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail="The p-series convergence criterion proves divergence.",
                notes=(BackendMathNote(label="Convergence condition", expression=convergence),),
            ),
        )
    zeta_value = sp.zeta(sp.Integer(power), evaluate=False)
    first_step = BackendDerivationStep(
        rule="Recognize a convergent p-series",
        before=sigma,
        after=zeta_value,
        explanation=f"This is a p-series with p = {power}, so it equals zeta({power}).",
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail="The p-series identity applies because the lower index is 1.",
        notes=(
            BackendMathNote(label="P-series identity", expression=identity),
            BackendMathNote(label="Convergence condition", expression=convergence),
        ),
    )
    if zeta_value == result:
        return (first_step,)
    return (
        first_step,
        BackendDerivationStep(
            rule=f"Use the exact value of zeta({power})",
            before=zeta_value,
            after=result,
            explanation="Substitute the known exact zeta value and simplify.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The special zeta value matches the exact symbolic result.",
            notes=(
                BackendMathNote(
                    label="Exact zeta value",
                    expression=BackendIdentity(left=zeta_value, right=result),
                ),
            ),
        ),
    )


def _derive_shifted_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    simplified = sp.simplify(expression)
    if upper != sp.oo or not simplified.is_Pow or len(simplified.args) != _GEOMETRIC_POWER_ARITY:
        return ()
    base, exponent = simplified.args
    power = next(
        (candidate for candidate in range(1, 13) if exponent == -sp.Integer(candidate)),
        None,
    )
    if power is None:
        return ()
    shift = sp.simplify(variable - base)
    if shift == sp.Integer(0) or variable in shift.free_symbols:
        return ()
    if sp.simplify(lower - shift) != sp.Integer(1):
        return ()
    shifted_variable = sp.Symbol("k", integer=True, positive=True)
    shifted_expression = shifted_variable ** (-power)
    shifted_sigma = BackendSigma(
        expression=shifted_expression,
        variable=shifted_variable,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    shift_step = BackendDerivationStep(
        rule="Shift the summation index",
        before=BackendSigma(
            expression=expression,
            variable=variable,
            lower=lower,
            upper=upper,
        ),
        after=shifted_sigma,
        explanation=(
            "Choose a new index that removes the constant shift. "
            "The lower bound becomes 1 and the upper bound remains infinity."
        ),
        verification_method=VerificationMethod.SUBSTITUTION,
        verification_detail="The index substitution preserves every term and both bounds.",
        notes=(
            BackendMathNote(
                label="Index substitution",
                expression=BackendIdentity(left=shifted_variable, right=base),
            ),
        ),
    )
    return (
        shift_step,
        *_derive_p_series(
            shifted_expression,
            shifted_variable,
            sp.Integer(1),
            sp.oo,
            result,
        ),
    )


def _derive_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if (
        lower.is_integer is not True
        or not expression.is_Pow
        or len(expression.args) != _GEOMETRIC_POWER_ARITY
    ):
        return ()
    ratio, exponent = expression.args
    if exponent != variable:
        return ()
    generic_ratio = sp.Symbol("r", real=True)
    generic_lower = sp.Symbol("m", integer=True, nonnegative=True)
    identity_lower = sp.Integer(0) if lower == sp.Integer(0) else generic_lower
    condition: BackendExpression
    if upper == sp.oo:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=generic_ratio**variable,
                variable=variable,
                lower=identity_lower,
                upper=sp.oo,
            ),
            right=generic_ratio**identity_lower / (1 - generic_ratio),
        )
        condition = sp.Lt(sp.Abs(generic_ratio), sp.Integer(1))
        ratio_magnitude = sp.Abs(ratio)
        converges = sp.simplify(sp.Lt(ratio_magnitude, sp.Integer(1)))
        if converges is sp.true:
            explanation = "The common ratio satisfies |r| < 1, so the series converges."
            rule = "Apply the infinite geometric-series identity"
            after: BackendExpression = result
            verification_detail = "The geometric-series formula gives the exact result."
        else:
            explanation = (
                "An infinite geometric series converges only when |r| < 1. "
                "This common ratio does not satisfy that condition."
            )
            rule = "Apply the geometric-series convergence test"
            after = sp.Ge(ratio_magnitude, sp.Integer(1), evaluate=False)
            verification_detail = "The convergence criterion was applied to the common ratio."
    elif upper.is_integer is True:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=generic_ratio**variable,
                variable=variable,
                lower=identity_lower,
                upper=upper,
            ),
            right=(generic_ratio**identity_lower - generic_ratio ** (upper + 1))
            / (1 - generic_ratio),
        )
        condition = BackendNotEqual(left=generic_ratio, right=sp.Integer(1))
        explanation = "Use the finite geometric-series identity and substitute the ratio."
        rule = "Apply the finite geometric-series identity"
        after = result
        verification_detail = "The geometric-series formula gives the exact result."
    else:
        return ()
    return (
        BackendDerivationStep(
            rule=rule,
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=after,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=verification_detail,
            notes=(
                BackendMathNote(label="Identity", expression=identity),
                BackendMathNote(label="Required condition", expression=condition),
                BackendMathNote(
                    label="Common ratio",
                    expression=BackendIdentity(left=generic_ratio, right=ratio),
                ),
                *(
                    (
                        BackendMathNote(
                            label="Starting index",
                            expression=BackendIdentity(left=generic_lower, right=lower),
                        ),
                    )
                    if lower != sp.Integer(0)
                    else ()
                ),
            ),
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
        _derive_p_series,
        _derive_shifted_p_series,
        _derive_geometric_series,
    ):
        derivation = strategy(expression, variable, lower, upper, result)
        if derivation:
            return derivation
    return ()
