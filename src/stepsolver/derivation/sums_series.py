"""Closed-form p-series, geometric, and telescoping strategies."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendExpression,
    BackendIdentity,
    BackendMathNote,
    BackendNotEqual,
    BackendSigma,
)
from stepsolver.results import VerificationMethod

_SQUARE_POWER = 2
_GEOMETRIC_POWER_ARITY = 2


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


def _derive_even_odd_p_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    _result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the even or odd residue class of a convergent p-series."""
    if upper != sp.oo:
        return ()
    even_base = 2 * variable
    odd_base = 2 * variable + 1
    match = next(
        (
            (family, power, coefficient)
            for family, base, expected_lower in (
                ("even", even_base, sp.Integer(1)),
                ("odd", odd_base, sp.Integer(0)),
            )
            for power in range(2, 13)
            if lower == expected_lower
            and not (coefficient := sp.simplify(expression * base**power)).has(variable)
            and coefficient != sp.Integer(0)
        ),
        None,
    )
    if match is None:
        return ()
    family, power, coefficient = match
    residue_factor = (
        sp.Rational(1, 2**power) if family == "even" else sp.Integer(1) - sp.Rational(1, 2**power)
    )
    factor = sp.simplify(coefficient * residue_factor)
    explanation = (
        "Factor 2 from every even denominator, then use the p-series identity."
        if family == "even"
        else "Subtract the even-indexed p-series from the full positive p-series."
    )
    zeta_value = sp.zeta(sp.Integer(power))
    expected = sp.simplify(factor * zeta_value)
    index = sp.Symbol("k", integer=True, positive=True)
    full_series = BackendSigma(
        expression=index ** (-power),
        variable=index,
        lower=sp.Integer(1),
        upper=sp.oo,
    )
    return (
        BackendDerivationStep(
            rule=f"Extract the {family} terms of the p-series",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=expected,
            explanation=explanation,
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The residue-class decomposition of the p-series is exact.",
            notes=(
                BackendMathNote(
                    label="Full p-series",
                    expression=BackendIdentity(left=full_series, right=zeta_value),
                ),
                BackendMathNote(
                    label=f"{family.title()}-term factor",
                    expression=BackendIdentity(
                        left=sp.Symbol("c", real=True),
                        right=factor,
                    ),
                ),
            ),
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
    if exponent_slope.has(variable) or coefficient.has(variable) or converges is not sp.true:
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


def _derive_second_moment_geometric_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Sum ``c*n**2*r**n`` by differentiating the geometric identity twice."""
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
    coefficient = sp.simplify(expression / (variable**2 * ratio**variable))
    if (
        exponent_slope.has(variable)
        or coefficient.has(variable)
        or sp.simplify(sp.Lt(sp.Abs(ratio), sp.Integer(1))) is not sp.true
    ):
        return ()
    formula = sp.simplify(coefficient * ratio * (1 + ratio) / (1 - ratio) ** 3)
    if formula != result:
        return ()
    generic_ratio = sp.Symbol("r", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=variable**2 * generic_ratio**variable,
            variable=variable,
            lower=sp.Integer(1),
            upper=sp.oo,
        ),
        right=generic_ratio * (1 + generic_ratio) / (1 - generic_ratio) ** 3,
    )
    return (
        BackendDerivationStep(
            rule="Differentiate the geometric-series identity twice",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation=(
                "Differentiate the geometric series twice, then combine the resulting "
                "first- and second-derivative terms and substitute the common ratio."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The second-moment geometric identity gives the exact sum.",
            notes=(
                BackendMathNote(label="Second-moment identity", expression=identity),
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


def _has_constant_numerator_quadratic_denominator(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> bool:
    """Return whether a term can have two shifted linear denominator factors."""
    numerator, denominator = sp.fraction(sp.cancel(expression))
    if numerator.has(variable):
        return False
    try:
        denominator_polynomial = sp.Poly(denominator, variable)
    except sp.PolynomialError:
        return False
    return denominator_polynomial.degree() == _SQUARE_POWER


def _derive_telescoping_rational_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use partial fractions for c/(n(n+k)) and cancel interior terms."""
    if lower.is_integer is not True or not _has_constant_numerator_quadratic_denominator(
        expression, variable
    ):
        return ()
    match = next(
        (
            (offset, shift, coefficient)
            for offset in range(-6, 7)
            for shift in range(1, 7)
            if (
                coefficient := sp.simplify(
                    expression * (variable + offset) * (variable + offset + shift)
                )
            )
            != sp.Integer(0)
            and not coefficient.has(variable)
            and sp.simplify(lower + offset).is_positive is True
        ),
        None,
    )
    if match is None:
        return ()
    offset, shift, coefficient = match
    shifted_variable = variable + offset
    decomposition = coefficient / shift * (1 / shifted_variable - 1 / (shifted_variable + shift))
    if sp.simplify(decomposition - expression) != sp.Integer(0):
        return ()
    leading_terms = sp.Add(
        *(1 / (lower + offset + index) for index in range(shift)),
    )
    match upper:
        case value if value == sp.oo:
            boundary_value = sp.simplify(coefficient * leading_terms / shift)
        case value if value.is_integer is True:
            trailing_terms = sp.Add(
                *(1 / (value + 1 + offset + index) for index in range(shift)),
            )
            boundary_value = sp.simplify(coefficient * (leading_terms - trailing_terms) / shift)
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


def _derive_gregory_leibniz_series(
    expression: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use the arctangent power series at x = 1."""
    expected_term = (-1) ** variable / (2 * variable + 1)
    if (
        lower != sp.Integer(0)
        or upper != sp.oo
        or sp.simplify(expression - expected_term) != sp.Integer(0)
        or sp.simplify(result - sp.pi / 4) != sp.Integer(0)
    ):
        return ()
    generic = sp.Symbol("x", real=True)
    identity = BackendIdentity(
        left=BackendSigma(
            expression=(-1) ** variable * generic ** (2 * variable + 1) / (2 * variable + 1),
            variable=variable,
            lower=sp.Integer(0),
            upper=sp.oo,
        ),
        right=sp.atan(generic),
    )
    return (
        BackendDerivationStep(
            rule="Apply the Gregory-Leibniz arctangent series",
            before=BackendSigma(
                expression=expression,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=result,
            explanation="Set x = 1 in the arctangent power series, so arctan(1) = pi/4.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The convergent endpoint value of the arctangent series applies.",
            notes=(BackendMathNote(label="Arctangent series", expression=identity),),
        ),
    )


CLOSED_FORM_SUM_STRATEGIES = (
    _derive_p_series,
    _derive_shifted_p_series,
    _derive_even_odd_p_series,
    _derive_geometric_series,
    _derive_differentiated_geometric_series,
    _derive_second_moment_geometric_series,
    _derive_telescoping_rational_sum,
    _derive_gregory_leibniz_series,
)
