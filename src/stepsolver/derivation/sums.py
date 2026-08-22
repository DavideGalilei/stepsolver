"""Human-readable derivations and domain checks for summations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDifference,
    BackendEvaluationAtIndex,
    BackendExpression,
    BackendIdentity,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
    BackendNthRoot,
    BackendProduct,
    BackendQuotient,
    BackendSigma,
    BackendStepConstraint,
    BackendSum,
    BackendUndefined,
)
from stepsolver.results import VerificationMethod
from stepsolver.sympy_series import match_alternating_p_series, match_harmonic_sine_series
from stepsolver.sympy_support import contains_unevaluated_operation, is_object_sequence

_SQUARE_POWER = 2
_GEOMETRIC_POWER_ARITY = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class UndefinedSummation:
    """One included index at which a summand is undefined."""

    index: sp.Basic
    denominator: sp.Basic
    steps: tuple[BackendDerivationStep, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class NthRootPowerMatch:
    """A positive base raised to and rooted by the summation index."""

    root_term: sp.Basic
    radicand: sp.Basic
    base: sp.Basic
    coefficient: sp.Basic


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


def _factorial_series_base(expression: sp.Basic, variable: sp.Symbol) -> sp.Basic | None:
    """Match ``base**n / n!`` with a constant base."""
    numerator = sp.simplify(expression * sp.factorial(variable))
    if numerator == sp.Integer(1):
        return sp.Integer(1)
    if not numerator.is_Pow or len(numerator.args) != _GEOMETRIC_POWER_ARITY:
        return None
    base, exponent = numerator.args
    if exponent != variable or base.has(variable):
        return None
    return base


def _derive_factorial_series_tail(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a tail of the exponential power series with explicit omitted terms."""
    if (
        upper != sp.oo
        or lower.is_integer is not True
        or lower.is_nonnegative is not True
    ):
        return ()
    base = _factorial_series_base(expression, variable)
    if base is None:
        return ()
    full_value = sp.exp(base)
    generic_base = sp.Symbol("x", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=generic_base**variable / sp.factorial(variable),
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        ),
        right=sp.exp(generic_base),
    )
    tail = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if lower == sp.Integer(0):
        if sp.simplify(full_value - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule="Apply the exponential-series identity",
                before=tail,
                after=full_value,
                explanation="This is the Maclaurin series for the exponential function.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail="The exponential power-series identity applies exactly.",
                notes=(BackendMathNote(label="Identity", expression=identity),),
            ),
        )

    prefix_upper = lower - 1
    prefix_sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=sp.Integer(0),
        upper=prefix_upper,
    )
    prefix_value = sp.summation(expression, (variable, sp.Integer(0), prefix_upper))
    expected = sp.simplify(full_value - prefix_value)
    if sp.simplify(expected - result) != sp.Integer(0):
        return ()
    rewritten_tail = BackendDifference(left=full_value, right=prefix_sigma)
    evaluated_tail = BackendDifference(left=full_value, right=prefix_value)
    return (
        BackendDerivationStep(
            rule="Subtract the omitted exponential-series terms",
            before=tail,
            after=rewritten_tail,
            explanation=(
                "Start with the full exponential series at n = 0, then subtract every term "
                f"before n = {lower}."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="Removing the finite prefix leaves exactly the requested tail.",
            notes=(BackendMathNote(label="Exponential-series identity", expression=identity),),
        ),
        BackendDerivationStep(
            rule="Evaluate the omitted finite terms",
            before=rewritten_tail,
            after=evaluated_tail,
            explanation=(
                "Evaluate the factorials in the finite prefix and combine those rational terms."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The finite prefix was evaluated exactly.",
            notes=(
                BackendMathNote(
                    label="Finite prefix",
                    expression=BackendIdentity(left=prefix_sigma, right=prefix_value),
                ),
            ),
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


def _derive_differentiated_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Sum c*n*r^n by differentiating the geometric-series identity."""
    if lower != sp.Integer(1) or upper != sp.oo:
        return ()
    power_match = next(
        (
            match
            for factor in expression.as_ordered_factors()
            if (match := _variable_power(factor, variable)) is not None
        ),
        None,
    )
    if power_match is None:
        return ()
    exponential_base, exponential_exponent = power_match
    exponent_slope = sp.diff(exponential_exponent, variable)
    ratio = sp.simplify(exponential_base**exponent_slope)
    coefficient = sp.simplify(expression / (variable * ratio**variable))
    converges = sp.simplify(sp.Lt(sp.Abs(ratio), sp.Integer(1)))
    if (
        exponent_slope.has(variable)
        or coefficient.has(variable)
        or converges is not sp.true
    ):
        return ()
    formula = sp.simplify(coefficient * ratio / (1 - ratio) ** 2)
    if formula != result:
        return ()
    generic_ratio = sp.Symbol("r", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=variable * generic_ratio**variable,
            variable=variable,
            lower=sp.Integer(1),
            upper=sp.oo,
        ),
        right=generic_ratio / (1 - generic_ratio) ** 2,
    )
    return (
        BackendDerivationStep(
            rule="Differentiate the geometric-series identity",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation=(
                "Differentiate Σrⁿ = 1/(1-r), multiply by r, then substitute the common "
                "ratio and outside coefficient."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The differentiated geometric identity gives the exact sum.",
            notes=(
                BackendMathNote(label="Weighted geometric identity", expression=identity),
                BackendMathNote(
                    label="Common ratio",
                    expression=BackendIdentity(left=generic_ratio, right=ratio),
                ),
                BackendMathNote(
                    label="Convergence condition",
                    expression=sp.Lt(sp.Abs(generic_ratio), sp.Integer(1)),
                ),
            ),
        ),
    )


def _variable_power(
    factor: sp.Basic,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic] | None:
    if not factor.is_Pow or len(factor.args) != _GEOMETRIC_POWER_ARITY:
        return None
    base, exponent = factor.args
    if exponent.has(variable):
        return base, exponent
    if base.is_Pow and len(base.args) == _GEOMETRIC_POWER_ARITY:
        nested_base, nested_exponent = base.args
        combined_exponent = sp.simplify(nested_exponent * exponent)
        if combined_exponent.has(variable):
            return nested_base, combined_exponent
    return None


def _derive_telescoping_rational_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use partial fractions for c/(n(n+k)) and cancel interior terms."""
    if lower.is_integer is not True or lower.is_positive is not True:
        return ()
    match = next(
        (
            (shift, sp.simplify(expression * variable * (variable + shift)))
            for shift in range(1, 7)
            if not sp.simplify(expression * variable * (variable + shift)).has(variable)
        ),
        None,
    )
    if match is None or match[1] == sp.Integer(0):
        return ()
    shift, coefficient = match
    decomposition = coefficient / shift * (
        1 / variable - 1 / (variable + shift)
    )
    if sp.simplify(decomposition - expression) != sp.Integer(0):
        return ()
    leading_terms = sp.Add(
        *(1 / (lower + offset) for offset in range(shift)),
    )
    match upper:
        case value if value == sp.oo:
            boundary_value = sp.simplify(coefficient * leading_terms / shift)
        case value if value.is_integer is True:
            trailing_terms = sp.Add(
                *(1 / (value + 1 + offset) for offset in range(shift)),
            )
            boundary_value = sp.simplify(
                coefficient * (leading_terms - trailing_terms) / shift
            )
        case _:
            return ()
    if boundary_value != result:
        return ()
    decomposed_sigma = BackendSigma(
        expression=decomposition,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    return (
        BackendDerivationStep(
            rule="Decompose the summand into partial fractions",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=decomposed_sigma,
            explanation="Split the rational term into shifted reciprocals with opposite signs.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the partial fractions recovers the original summand.",
            notes=(
                BackendMathNote(
                    label="Partial-fraction identity",
                    expression=BackendIdentity(left=expression, right=decomposition),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Cancel the telescoping terms",
            before=decomposed_sigma,
            after=result,
            explanation=(
                "Write the partial sums: every interior reciprocal cancels, leaving only the "
                "boundary terms."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The surviving boundary terms simplify to the exact result.",
            notes=(
                BackendMathNote(
                    label="Surviving boundary value",
                    expression=BackendIdentity(left=boundary_value, right=result),
                ),
            ),
        ),
    )


def _scaled_expression(
    amplitude: sp.Basic,
    expression: BackendExpression,
) -> BackendExpression:
    if amplitude == sp.Integer(1):
        return expression
    return BackendProduct(factors=(amplitude, expression))


def _derive_harmonic_sine_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    match = match_harmonic_sine_series(expression, variable, lower, upper)
    if match is None or match.value != result:
        return ()
    sigma = BackendSigma(
        expression=expression,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    if match.normalized_angle == sp.Integer(0):
        return (
            BackendDerivationStep(
                rule="Simplify the sine terms",
                before=sigma,
                after=sp.Integer(0),
                explanation=(
                    "The angle is a whole number of full turns, so every sine term is zero."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="Each summand simplifies exactly to zero.",
            ),
        )

    damping = sp.Symbol("r", real=True, positive=True)
    generic_angle = sp.Symbol("theta", real=True, positive=True)
    partial_index = sp.Symbol("k", integer=True, positive=True)
    partial_upper = sp.Symbol("N", integer=True, positive=True)
    damped_sigma = BackendSigma(
        expression=(
            match.amplitude
            * damping**variable
            * sp.sin(variable * match.normalized_angle)
            / variable
        ),
        variable=variable,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    damped_limit = BackendLimit(
        expression=damped_sigma,
        variable=damping,
        point=sp.Integer(1),
        direction="-",
    )
    arctangent = _scaled_expression(
        match.amplitude,
        sp.atan(
            damping * sp.sin(match.normalized_angle)
            / (1 - damping * sp.cos(match.normalized_angle))
        ),
    )
    arctangent_limit = BackendLimit(
        expression=arctangent,
        variable=damping,
        point=sp.Integer(1),
        direction="-",
    )
    sawtooth_value = _scaled_expression(
        match.amplitude,
        BackendQuotient(
            numerator=BackendDifference(left=sp.pi, right=match.normalized_angle),
            denominator=sp.Integer(2),
        ),
    )
    return (
        BackendDerivationStep(
            rule="Introduce an Abel convergence factor",
            before=sigma,
            after=damped_limit,
            explanation=(
                "First reduce the angle modulo 2*pi, which does not change any sine term. "
                "The factors 1/n decrease to zero, while the partial sums of the sine terms "
                "stay bounded. Dirichlet's test proves convergence, and Abel's theorem lets "
                "us approach the series through the easier damped series with 0 < r < 1."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Dirichlet's test and Abel's limit theorem apply to the harmonic sine series."
            ),
            notes=(
                BackendMathNote(
                    label="Coefficient limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=1 / variable,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
                BackendMathNote(
                    label="Normalized angle",
                    expression=BackendIdentity(
                        left=generic_angle,
                        right=match.normalized_angle,
                    ),
                ),
                BackendMathNote(
                    label="Sine partial-sum identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=sp.sin(partial_index * generic_angle),
                            variable=partial_index,
                            lower=sp.Integer(1),
                            upper=partial_upper,
                        ),
                        right=(
                            sp.sin(partial_upper * generic_angle / 2)
                            * sp.sin((partial_upper + 1) * generic_angle / 2)
                            / sp.sin(generic_angle / 2)
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Sum the damped sine series",
            before=damped_limit,
            after=arctangent_limit,
            explanation=(
                "For 0 < r < 1, integrate the geometric series term by term and take its "
                "imaginary part. This gives an arctangent closed form."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The damped identity follows from the absolutely convergent geometric series."
            ),
            notes=(
                BackendMathNote(
                    label="Damped sine-series identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=(
                                damping**variable
                                * sp.sin(variable * generic_angle)
                                / variable
                            ),
                            variable=variable,
                            lower=sp.Integer(1),
                            upper=sp.oo,
                        ),
                        right=sp.atan(
                            damping
                            * sp.sin(generic_angle)
                            / (1 - damping * sp.cos(generic_angle))
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Remove the convergence factor",
            before=arctangent_limit,
            after=sawtooth_value,
            explanation=(
                "Let r approach 1 from below. For an angle between 0 and 2*pi, the "
                "arctangent limit is (pi - theta)/2."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The one-sided limit gives the Fourier sawtooth value.",
            notes=(
                BackendMathNote(
                    label="Harmonic sine-series identity",
                    expression=BackendIdentity(
                        left=BackendSigma(
                            expression=sp.sin(variable * generic_angle) / variable,
                            variable=variable,
                            lower=sp.Integer(1),
                            upper=sp.oo,
                        ),
                        right=BackendQuotient(
                            numerator=BackendDifference(left=sp.pi, right=generic_angle),
                            denominator=sp.Integer(2),
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Simplify the exact value",
            before=sawtooth_value,
            after=result,
            explanation="Substitute the normalized angle and simplify the exact arithmetic.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The displayed expression simplifies to the exact result.",
        ),
    )


def _derive_alternating_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    match = match_alternating_p_series(expression, variable, lower, upper)
    if match is None or match.value != result:
        return ()
    partial_upper = sp.Symbol("N", integer=True, positive=True)
    starting_coefficient = -match.coefficient
    canonical_expression = (
        starting_coefficient * (-sp.Integer(1)) ** (variable + 1) / variable**match.power
    )
    partial_sum_limit = BackendLimit(
        expression=BackendSigma(
            expression=canonical_expression,
            variable=variable,
            lower=sp.Integer(1),
            upper=partial_upper,
        ),
        variable=partial_upper,
        point=sp.oo,
    )
    magnitude = variable ** (-match.power)
    first_step = BackendDerivationStep(
        rule="Apply the Leibniz alternating-series test",
        before=BackendSigma(
            expression=expression,
            variable=variable,
            lower=lower,
            upper=upper,
        ),
        after=partial_sum_limit,
        explanation=(
            "Factor out the fixed overall coefficient. The remaining term signs alternate, "
            "and their positive magnitudes decrease to zero. Therefore the partial sums "
            "converge by the Leibniz test."
        ),
        verification_method=VerificationMethod.BACKEND_IDENTITY,
        verification_detail=(
            "The positivity, monotone decrease, and zero-limit conditions were checked."
        ),
        notes=(
            BackendMathNote(
                label="Core magnitude limit",
                expression=BackendIdentity(
                    left=BackendLimit(
                        expression=magnitude,
                        variable=variable,
                        point=sp.oo,
                    ),
                    right=sp.Integer(0),
                ),
            ),
            BackendMathNote(
                label="Decreasing core magnitudes",
                expression=sp.Lt(
                    (variable + 1) ** (-match.power),
                    magnitude,
                ),
            ),
        ),
    )
    generic_power = sp.Symbol("p", real=True, positive=True)
    if match.power == 1:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=(-sp.Integer(1)) ** (variable + 1) / variable,
                variable=variable,
                lower=sp.Integer(1),
                upper=sp.oo,
            ),
            right=sp.log(sp.Integer(2)),
        )
        rule = "Use the alternating harmonic-series identity"
        explanation = "The convergent alternating harmonic series has exact value log(2)."
    else:
        identity = BackendIdentity(
            left=BackendSigma(
                expression=(-sp.Integer(1)) ** (variable + 1) / variable**generic_power,
                variable=variable,
                lower=sp.Integer(1),
                upper=sp.oo,
            ),
            right=(1 - sp.Integer(2) ** (1 - generic_power)) * sp.zeta(generic_power),
        )
        rule = "Use the Dirichlet eta identity"
        explanation = (
            "Relate the convergent alternating p-series to the Riemann zeta function, "
            f"then substitute p = {match.power} and the overall coefficient."
        )
    return (
        first_step,
        BackendDerivationStep(
            rule=rule,
            before=partial_sum_limit,
            after=result,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The standard alternating-series identity gives the exact value.",
            notes=(BackendMathNote(label="Exact identity", expression=identity),),
        ),
    )


def _derive_nth_term_divergence(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if upper != sp.oo or lower.is_integer is not True:
        return ()
    try:
        term_limit = sp.limit(expression, variable, sp.oo)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if contains_unevaluated_operation(term_limit):
        return ()
    is_nonzero = sp.simplify(sp.Ne(term_limit, sp.Integer(0)))
    if is_nonzero is not sp.true:
        return ()
    return (
        BackendDerivationStep(
            rule="Apply the nth-term divergence test",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation=(
                "A necessary condition for an infinite series to converge is that its terms "
                "approach zero. Here the term limit is nonzero, so the series diverges."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The summand's limit at infinity is not zero.",
            notes=(
                BackendMathNote(
                    label="Term limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=expression,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=term_limit,
                    ),
                ),
            ),
        ),
    )


def _match_nth_root_power(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> NthRootPowerMatch | None:
    """Find one term of the form ``(c * base**n)**(1/n)`` with positive factors."""
    for term in expression.as_ordered_terms():
        if not term.is_Pow or len(term.args) != _GEOMETRIC_POWER_ARITY:
            continue
        radicand, reciprocal_index = term.args
        if sp.simplify(reciprocal_index - 1 / variable) != sp.Integer(0):
            continue
        for factor in radicand.as_ordered_factors():
            if not factor.is_Pow or len(factor.args) != _GEOMETRIC_POWER_ARITY:
                continue
            base, exponent = factor.args
            coefficient = sp.simplify(radicand / factor)
            if (
                exponent == variable
                and base.is_positive is True
                and not coefficient.has(variable)
                and coefficient.is_positive is True
            ):
                return NthRootPowerMatch(
                    root_term=term,
                    radicand=radicand,
                    base=base,
                    coefficient=coefficient,
                )
    return None


def _nth_root_simplification(
    match: NthRootPowerMatch,
    variable: sp.Symbol,
) -> tuple[sp.Basic, BackendExpression]:
    """Return backend and human forms of ``base * root(c, n)``."""
    if match.coefficient == sp.Integer(1):
        return match.base, match.base
    reciprocal_index = sp.Pow(variable, -1, evaluate=False)
    coefficient_root = sp.Pow(match.coefficient, reciprocal_index, evaluate=False)
    backend_expression = sp.Mul(match.base, coefficient_root, evaluate=False)
    displayed_expression = BackendProduct(
        factors=(
            match.base,
            BackendNthRoot(radicand=match.coefficient, index=variable),
        ),
    )
    return backend_expression, displayed_expression


def _nth_term_divergence_conclusion(term_limit: sp.Basic) -> sp.Basic | None:
    """Choose the direction forced by an eventually signed nonzero term limit."""
    if term_limit == sp.oo or term_limit.is_positive is True:
        return sp.oo
    if term_limit == -sp.oo or term_limit.is_negative is True:
        return -sp.oo
    return None


def _derive_nth_root_power_divergence(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    _result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Simplify an indexed root of a matching positive power before testing divergence."""
    if upper != sp.oo or lower.is_integer is not True or lower.is_positive is not True:
        return ()
    match = _match_nth_root_power(expression, variable)
    if match is None:
        return ()
    simplified_root, displayed_simplified_root = _nth_root_simplification(match, variable)
    remaining = sp.simplify(expression - match.root_term)
    simplified_expression = sp.simplify(simplified_root + remaining)
    try:
        term_limit = sp.limit(simplified_expression, variable, sp.oo)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if contains_unevaluated_operation(term_limit):
        return ()
    divergence_conclusion = _nth_term_divergence_conclusion(term_limit)
    if divergence_conclusion is None:
        return ()

    displayed_root = BackendNthRoot(radicand=match.radicand, index=variable)
    displayed_original: BackendExpression = displayed_root
    displayed_simplified = displayed_simplified_root
    if remaining != sp.Integer(0):
        displayed_original = BackendSum(terms=(displayed_root, remaining))
        displayed_simplified = BackendSum(terms=(displayed_simplified_root, remaining))
    original_sigma = BackendSigma(
        expression=displayed_original,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    simplified_sigma = BackendSigma(
        expression=displayed_simplified,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    return (
        BackendDerivationStep(
            rule="Simplify the indexed root",
            before=original_sigma,
            after=simplified_sigma,
            explanation=(
                "Separate the matching positive n-th power from the other positive "
                "factors. Its n-th root equals the base."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The indexed root and power have the same positive index.",
            notes=(
                BackendMathNote(
                    label="Root-power identity",
                    expression=BackendIdentity(
                        left=displayed_root,
                        right=displayed_simplified_root,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Apply the nth-term divergence test",
            before=simplified_sigma,
            after=divergence_conclusion,
            explanation=(
                "A convergent infinite series must have terms approaching zero. "
                "The simplified summand instead has a nonzero limit."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The simplified summand's limit at infinity is not zero.",
            notes=(
                BackendMathNote(
                    label="Term limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=displayed_simplified,
                            variable=variable,
                            point=sp.oo,
                        ),
                        right=term_limit,
                    ),
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
        _derive_factorial_series_tail,
        _derive_p_series,
        _derive_shifted_p_series,
        _derive_geometric_series,
        _derive_differentiated_geometric_series,
        _derive_telescoping_rational_sum,
        _derive_harmonic_sine_series,
        _derive_alternating_p_series,
        _derive_nth_root_power_divergence,
        _derive_nth_term_divergence,
    ):
        derivation = strategy(expression, variable, lower, upper, result)
        if derivation:
            return derivation
    return ()
