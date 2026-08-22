"""Human-readable finite and infinite limit strategies."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendInlineMath,
    BackendLimit,
    BackendMathNote,
    BackendNotEqual,
    BackendProduct,
    BackendQuotient,
)
from stepsolver.results import VerificationMethod

_POWER_ARITY = 2


def _derive_factorial_recurrence_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Cancel neighboring factorials before evaluating an infinite limit."""
    if point not in {sp.oo, -sp.oo} or not expression.has(sp.factorial(variable)):
        return ()
    simplified = sp.simplify(expression)
    if simplified == expression or simplified.has(sp.factorial):
        return ()
    try:
        simplified_result = sp.limit(simplified, variable, point)
    except (NotImplementedError, TypeError, ValueError):
        return ()
    if sp.simplify(simplified_result - result) != sp.Integer(0):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    simplified_limit = BackendLimit(
        expression=simplified,
        variable=variable,
        point=point,
        direction=direction,
    )
    generic = sp.Symbol("k", integer=True, positive=True)
    recurrence = BackendIdentity(
        left=sp.factorial(generic + 1),
        right=(generic + 1) * sp.factorial(generic),
    )
    return (
        BackendDerivationStep(
            rule="Apply the factorial recurrence",
            before=displayed,
            after=simplified_limit,
            explanation="Expand the larger neighboring factorial, then cancel the common one.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The factorial recurrence gives the simplified quotient exactly.",
            notes=(BackendMathNote(label="Factorial recurrence", expression=recurrence),),
        ),
        BackendDerivationStep(
            rule="Evaluate the simplified rational limit",
            before=simplified_limit,
            after=result,
            explanation="Compare the remaining leading powers as the variable tends to infinity.",
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail="The simplified rational expression has the displayed limit.",
        ),
    )


def _derive_infinite_radical_conjugate(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Rationalize ``x*(sqrt(x**2+c)-x)`` at positive infinity."""
    if point != sp.oo:
        return ()
    difference = sp.simplify(expression / variable)
    radicand = sp.simplify((difference + variable) ** 2)
    constant = sp.simplify(radicand - variable**2)
    if constant.has(variable) or constant.is_positive is not True:
        return ()
    root = sp.sqrt(variable**2 + constant)
    if sp.simplify(difference - (root - variable)) != sp.Integer(0):
        return ()
    rationalized = constant * variable / (root + variable)
    normalized = constant / (sp.sqrt(1 + constant / variable**2) + 1)
    if sp.simplify(result - constant / 2) != sp.Integer(0):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    rationalized_limit = BackendLimit(
        expression=rationalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    normalized_limit = BackendLimit(
        expression=normalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Multiply by the conjugate",
            before=displayed,
            after=rationalized_limit,
            explanation="Use the conjugate so the difference of squares removes the subtraction.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Multiplying by the conjugate quotient preserves the expression.",
            notes=(BackendMathNote(label="Conjugate", expression=root + variable),),
        ),
        BackendDerivationStep(
            rule="Divide by the leading variable",
            before=rationalized_limit,
            after=normalized_limit,
            explanation=(
                "Divide numerator and denominator by the positive variable to expose terms "
                "that vanish at infinity."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The normalized quotient is equal for sufficiently large values.",
        ),
        BackendDerivationStep(
            rule="Evaluate the normalized limit",
            before=normalized_limit,
            after=result,
            explanation="The reciprocal-square term tends to zero, so substitute its limit.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Continuity of the square root gives the exact value.",
        ),
    )


def _constant_scale(
    expression: sp.Basic,
    reference: sp.Basic,
    variable: sp.Symbol,
) -> sp.Basic | None:
    """Return the nonzero constant relating expression to reference."""
    scale = sp.simplify(expression / reference)
    if scale == sp.Integer(0) or scale.has(variable):
        return None
    return scale


def _matching_terms(expression: sp.Basic, function: object) -> tuple[sp.Basic, ...]:
    current = (expression,) if expression.func == function else ()
    nested = tuple(
        item
        for argument in expression.args
        for item in _matching_terms(argument, function)
    )
    return (*current, *nested)


def _square_root_terms(expression: sp.Basic) -> tuple[sp.Basic, ...]:
    current = (
        (expression,)
        if expression.is_Pow
        and len(expression.args) == _POWER_ARITY
        and expression.args[1] == sp.Rational(1, 2)
        else ()
    )
    nested = tuple(
        item
        for argument in expression.args
        for item in _square_root_terms(argument)
    )
    return (*current, *nested)


def _derive_infinite_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    numerator, denominator = sp.fraction(sp.together(expression))
    try:
        numerator_polynomial = sp.Poly(numerator, variable)
        denominator_polynomial = sp.Poly(denominator, variable)
    except sp.PolynomialError:
        numerator_polynomial = None
        denominator_polynomial = None
    if numerator_polynomial is not None and denominator_polynomial is not None:
        numerator_degree = numerator_polynomial.degree()
        denominator_degree = denominator_polynomial.degree()
        if numerator_degree <= denominator_degree:
            numerator_leading = (
                numerator_polynomial.coeff_monomial(variable**numerator_degree)
                * variable**numerator_degree
            )
            denominator_leading = (
                denominator_polynomial.coeff_monomial(variable**denominator_degree)
                * variable**denominator_degree
            )
            return (
                BackendDerivationStep(
                    rule="Compare the leading powers",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "At infinity, the highest-degree terms determine the ratio's limiting "
                        "behavior."
                    ),
                    verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    verification_detail=(
                        "Lower-degree terms vanish relative to the leading powers."
                    ),
                    notes=(
                        BackendMathNote(
                            label="Leading-term ratio",
                            expression=BackendLimit(
                                expression=numerator_leading / denominator_leading,
                                variable=variable,
                                point=point,
                            ),
                        ),
                    ),
                ),
            )
    if expression.has(sp.exp(variable)) and result in {sp.oo, -sp.oo}:
        generic_exponent = sp.Symbol("n", positive=True)
        return (
            BackendDerivationStep(
                rule="Use exponential growth",
                before=displayed_limit,
                after=result,
                explanation="An exponential grows faster than every fixed power of the variable.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The standard exponential-over-power growth rule applies.",
                notes=(
                    BackendMathNote(
                        label="Growth rule",
                        expression=BackendIdentity(
                            left=BackendLimit(
                                expression=sp.exp(variable) / variable**generic_exponent,
                                variable=variable,
                                point=sp.oo,
                            ),
                            right=sp.oo,
                        ),
                    ),
                ),
            ),
        )
    return ()


def _derive_sine_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    numerator, denominator = sp.fraction(sp.together(expression))
    if point not in {sp.oo, -sp.oo} and numerator.func == sp.sin:
        offset = sp.simplify(variable - point)
        denominator_scale = _constant_scale(denominator, offset, variable)
        sine_argument = numerator.args[0]
        frequency = _constant_scale(sine_argument, offset, variable)
        if denominator_scale is not None and frequency is not None:
            expected = sp.simplify(frequency / denominator_scale)
        else:
            expected = None
        if expected is not None and sp.simplify(result - expected) == sp.Integer(0):
            generic_variable = sp.Symbol("u", real=True)
            is_standard = frequency == denominator_scale == sp.Integer(1)
            return (
                BackendDerivationStep(
                    rule=(
                        "Use the standard sine limit"
                        if is_standard
                        else "Normalize to the standard sine limit"
                    ),
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "Shift to the approach point, account for the constant angle and "
                        "denominator scales, then use the standard sine limit."
                    ),
                    verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                    verification_detail="The normalized quotient has the standard limit one.",
                    notes=(
                        BackendMathNote(
                            label="Rewrite the quotient",
                            expression=BackendIdentity(
                                left=expression,
                                right=BackendProduct(
                                    factors=(
                                        expected,
                                        BackendQuotient(
                                            numerator=sp.sin(sine_argument),
                                            denominator=sine_argument,
                                        ),
                                    )
                                ),
                            ),
                        ),
                        BackendMathNote(
                            label="Standard limit",
                            expression=BackendIdentity(
                                left=BackendLimit(
                                    expression=sp.sin(generic_variable) / generic_variable,
                                    variable=generic_variable,
                                    point=sp.Integer(0),
                                ),
                                right=sp.Integer(1),
                            ),
                        ),
                    ),
                ),
            )
    return ()


def _derive_standard_zero_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Recognize shifted/scaled exponential, logarithmic, and cosine limits."""
    if point in {sp.oo, -sp.oo}:
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    numerator, denominator = sp.fraction(sp.together(expression))
    offset = sp.simplify(variable - point)
    denominator_scale = _constant_scale(denominator, offset, variable)
    if denominator_scale is not None:
        exponential_terms = _matching_terms(numerator, sp.exp)
        if len(exponential_terms) == 1:
            exponential = exponential_terms[0]
            exponent_rate = _constant_scale(exponential.args[0], offset, variable)
            rate = (
                sp.simplify(exponent_rate / denominator_scale)
                if exponent_rate is not None
                else None
            )
            if (
                rate is not None
                and sp.simplify(numerator - (exponential - 1)) == sp.Integer(0)
                and sp.simplify(result - rate) == sp.Integer(0)
            ):
                generic = sp.Symbol("u", real=True)
                standard_limit = BackendIdentity(
                    left=BackendLimit(
                        expression=(sp.exp(generic) - 1) / generic,
                        variable=generic,
                        point=sp.Integer(0),
                    ),
                    right=sp.Integer(1),
                )
                return (
                    BackendDerivationStep(
                        rule="Normalize to the standard exponential limit",
                        before=displayed,
                        after=result,
                        explanation=(
                            "Use u equal to the exponent, factor out its constant rate, and "
                            "apply lim (eᵘ-1)/u = 1."
                        ),
                        explanation_parts=(
                            "Let ",
                            BackendInlineMath(expression=generic),
                            " equal the exponent, factor out its constant rate, and apply ",
                            BackendInlineMath(expression=standard_limit),
                            ".",
                        ),
                        verification_method=VerificationMethod.BACKEND_IDENTITY,
                        verification_detail="The normalized quotient is the standard limit.",
                        notes=(
                            BackendMathNote(
                                label="Standard exponential limit",
                                expression=standard_limit,
                            ),
                        ),
                    ),
                )
        if numerator.func == sp.log and len(numerator.args) == 1:
            logarithm_argument = numerator.args[0]
            increment_rate = _constant_scale(
                logarithm_argument - 1,
                offset,
                variable,
            )
            rate = (
                sp.simplify(increment_rate / denominator_scale)
                if increment_rate is not None
                else None
            )
            if (
                rate is not None
                and sp.simplify(result - rate) == sp.Integer(0)
            ):
                generic = sp.Symbol("u", real=True)
                standard_limit = BackendIdentity(
                    left=BackendLimit(
                        expression=sp.log(1 + generic) / generic,
                        variable=generic,
                        point=sp.Integer(0),
                    ),
                    right=sp.Integer(1),
                )
                return (
                    BackendDerivationStep(
                        rule="Normalize to the standard logarithm limit",
                        before=displayed,
                        after=result,
                        explanation=(
                            "Use u for the increment inside the logarithm, factor out its "
                            "constant rate, and apply lim log(1+u)/u = 1."
                        ),
                        explanation_parts=(
                            "Use ",
                            BackendInlineMath(expression=generic),
                            (
                                " for the increment inside the logarithm, factor out its "
                                "constant rate, and apply "
                            ),
                            BackendInlineMath(expression=standard_limit),
                            ".",
                        ),
                        verification_method=VerificationMethod.BACKEND_IDENTITY,
                        verification_detail="The normalized quotient is the standard limit.",
                        notes=(
                            BackendMathNote(
                                label="Standard logarithm limit",
                                expression=standard_limit,
                            ),
                        ),
                    ),
                )
    squared_denominator_scale = _constant_scale(
        denominator,
        offset**2,
        variable,
    )
    if squared_denominator_scale is not None:
        cosine_terms = _matching_terms(numerator, sp.cos)
        if len(cosine_terms) == 1:
            cosine = cosine_terms[0]
            rate = _constant_scale(cosine.args[0], offset, variable)
            expected = (
                sp.simplify(rate**2 / (2 * squared_denominator_scale))
                if rate is not None
                else None
            )
            if (
                expected is not None
                and sp.simplify(numerator - (1 - cosine)) == sp.Integer(0)
                and sp.simplify(result - expected) == sp.Integer(0)
            ):
                generic = sp.Symbol("u", real=True)
                half_angle_identity = BackendIdentity(
                    left=1 - sp.cos(generic),
                    right=2 * sp.sin(generic / 2) ** 2,
                )
                return (
                    BackendDerivationStep(
                        rule="Use the standard cosine limit",
                        before=displayed,
                        after=result,
                        explanation=(
                            "Normalize the angle and use 1-cos(u) = 2sin²(u/2), reducing the "
                            "limit to the standard sine limit."
                        ),
                        explanation_parts=(
                            "Normalize the angle and use ",
                            BackendInlineMath(expression=half_angle_identity),
                            ", reducing the limit to the standard sine limit.",
                        ),
                        verification_method=VerificationMethod.BACKEND_IDENTITY,
                        verification_detail="The half-angle identity gives the exact limit.",
                        notes=(
                            BackendMathNote(
                                label="Standard cosine limit",
                                expression=BackendIdentity(
                                    left=BackendLimit(
                                        expression=(1 - sp.cos(generic)) / generic**2,
                                        variable=generic,
                                        point=sp.Integer(0),
                                    ),
                                    right=sp.Rational(1, 2),
                                ),
                            ),
                        ),
                    ),
                )
    return ()


def _derive_radical_rationalization(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Rationalize a square-root difference that produces zero over zero."""
    rationalization = _radical_rationalization(expression, variable)
    if point in {sp.oo, -sp.oo} or rationalization is None:
        return ()
    conjugate, rationalized = rationalization
    substituted = sp.simplify(rationalized.subs(variable, point))
    if substituted != result or substituted.has(sp.zoo) or substituted.has(sp.nan):
        return ()
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    rationalized_limit = BackendLimit(
        expression=rationalized,
        variable=variable,
        point=point,
        direction=direction,
    )
    return (
        BackendDerivationStep(
            rule="Multiply by the conjugate",
            before=displayed,
            after=rationalized_limit,
            explanation=(
                "Multiply numerator and denominator by the conjugate, then use the difference "
                "of squares to remove the radical from the numerator."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="The conjugate quotient equals one away from the hole.",
            notes=(BackendMathNote(label="Conjugate", expression=conjugate),),
        ),
        BackendDerivationStep(
            rule="Substitute into the rationalized expression",
            before=rationalized_limit,
            after=result,
            explanation="The rationalized expression is continuous at the approach point.",
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Direct substitution gives the exact limit.",
        ),
    )


def _radical_rationalization(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic] | None:
    numerator, denominator = sp.fraction(sp.together(expression))
    square_roots = _square_root_terms(numerator)
    if len(square_roots) != 1:
        return None
    root = square_roots[0]
    constant = sp.simplify(root - numerator)
    if constant.has(variable) or constant == sp.Integer(0):
        return None
    if sp.simplify(numerator - (root - constant)) != sp.Integer(0):
        return None
    conjugate = root + constant
    radicand = root.args[0]
    rationalized = sp.cancel((radicand - constant**2) / (denominator * conjugate))
    if sp.simplify(rationalized - expression) != sp.Integer(0):
        return None
    return conjugate, rationalized


def _derive_variable_power_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Handle variable powers using the exponential of a logarithm."""
    displayed = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )
    if expression == variable**variable and point == sp.Integer(0) and direction == "+":
        rewritten = sp.exp(variable * sp.log(variable))
        exponent_limit = BackendLimit(
            expression=variable * sp.log(variable),
            variable=variable,
            point=point,
            direction=direction,
        )
        if result != sp.Integer(1):
            return ()
        return (
            BackendDerivationStep(
                rule="Rewrite the variable power exponentially",
                before=displayed,
                after=BackendLimit(
                    expression=rewritten,
                    variable=variable,
                    point=point,
                    direction=direction,
                ),
                explanation="For positive x, write xˣ as e^(x log(x)).",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The exponential-logarithm identity holds for x > 0.",
                notes=(BackendMathNote(label="Exponent limit", expression=exponent_limit),),
            ),
            BackendDerivationStep(
                rule="Evaluate the exponent limit",
                before=exponent_limit,
                after=result,
                explanation="Since x log(x) tends to zero, continuity of e^t gives e^0 = 1.",
                verification_method=VerificationMethod.BACKEND_IDENTITY,
                verification_detail=(
                    "The standard x log(x) limit and continuity of the exponential apply."
                ),
            ),
        )
    if point == sp.oo and expression.is_Pow and len(expression.args) == _POWER_ARITY:
        base, exponent = expression.args
        increment = sp.simplify((base - 1) * variable)
        exponent_scale = sp.simplify(exponent / variable)
        expected = sp.exp(sp.simplify(increment * exponent_scale))
        if (
            not increment.has(variable)
            and not exponent_scale.has(variable)
            and exponent_scale != sp.Integer(0)
            and sp.simplify(result - expected) == sp.Integer(0)
        ):
            generic = sp.Symbol("a", real=True)
            return (
                BackendDerivationStep(
                    rule="Use the exponential-definition limit",
                    before=displayed,
                    after=result,
                    explanation=(
                        "This has the form (1+a/x)^(b x), whose limit is e^(a b). "
                        "Substitute the constant increment and exponent scale."
                    ),
                    verification_method=VerificationMethod.BACKEND_IDENTITY,
                    verification_detail="The defining exponential limit gives the exact value.",
                    notes=(
                        BackendMathNote(
                            label="General identity",
                            expression=BackendIdentity(
                                left=BackendLimit(
                                    expression=(1 + generic / variable) ** variable,
                                    variable=variable,
                                    point=sp.oo,
                                ),
                                right=sp.exp(generic),
                            ),
                        ),
                    ),
                ),
            )
    return ()


def _derive_algebraic_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use cancellation, sign analysis, substitution, or growth comparison."""
    displayed_limit = BackendLimit(
        expression=expression,
        variable=variable,
        point=point,
        direction=direction,
    )

    numerator, denominator = sp.fraction(sp.together(expression))
    if point not in {sp.oo, -sp.oo}:
        numerator_at_point = sp.simplify(numerator.subs(variable, point))
        denominator_at_point = sp.simplify(denominator.subs(variable, point))
        if numerator_at_point == sp.Integer(0) and denominator_at_point == sp.Integer(0):
            canceled = sp.cancel(expression)
            if str(canceled) != str(expression):
                canceled_limit = BackendLimit(
                    expression=canceled,
                    variable=variable,
                    point=point,
                    direction=direction,
                )
                return (
                    BackendDerivationStep(
                        rule="Factor and cancel the common factor",
                        before=displayed_limit,
                        after=canceled_limit,
                        explanation=(
                            "Factor the numerator and denominator, then cancel the common "
                            "factor for nearby values."
                        ),
                        verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                        verification_detail=(
                            "The original and canceled expressions agree away from the hole."
                        ),
                        notes=(
                            BackendMathNote(
                                label="The limit ignores the value at the hole",
                                expression=BackendNotEqual(left=variable, right=point),
                            ),
                        ),
                    ),
                    BackendDerivationStep(
                        rule="Substitute into the simplified expression",
                        before=canceled_limit,
                        after=result,
                        explanation=(
                            "The simplified expression is continuous at the approach point, "
                            "so substitute directly."
                        ),
                        verification_method=VerificationMethod.EXACT_ARITHMETIC,
                        verification_detail="Direct substitution gives the exact limit.",
                    ),
                )

        if direction in {"+", "-"} and denominator == variable - point:
            expected = sp.oo if direction == "+" else -sp.oo
            if result == expected and numerator == sp.Integer(1):
                side_symbol = sp.Symbol("u", positive=True)
                signed_denominator = side_symbol if direction == "+" else -side_symbol
                return (
                    BackendDerivationStep(
                        rule="Analyze the sign from the requested side",
                        before=displayed_limit,
                        after=result,
                        explanation=(
                            "The denominator approaches zero through positive values."
                            if direction == "+"
                            else "The denominator approaches zero through negative values."
                        ),
                        verification_method=VerificationMethod.EXACT_ARITHMETIC,
                        verification_detail=(
                            "The reciprocal grows without bound with the indicated sign."
                        ),
                        notes=(
                            BackendMathNote(
                                label="Nearby denominator sign",
                                expression=BackendIdentity(
                                    left=denominator,
                                    right=signed_denominator,
                                ),
                            ),
                        ),
                    ),
                )

        substituted = sp.simplify(expression.subs(variable, point))
        if substituted == result and not substituted.has(sp.zoo) and not substituted.has(sp.nan):
            return (
                BackendDerivationStep(
                    rule="Use direct substitution",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "The expression is continuous at the approach point, so evaluate it "
                        "there directly."
                    ),
                    verification_method=VerificationMethod.EXACT_ARITHMETIC,
                    verification_detail="Exact substitution gives the displayed value.",
                ),
            )

    if point in {sp.oo, -sp.oo}:
        return _derive_infinite_limit(expression, variable, point, direction, result)
    return ()


def derive_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive familiar limits with the shortest standard student method."""
    strategies = (
        _derive_factorial_recurrence_limit,
        _derive_sine_limit,
        _derive_standard_zero_limit,
        _derive_variable_power_limit,
        _derive_radical_rationalization,
        _derive_infinite_radical_conjugate,
        _derive_algebraic_limit,
    )
    for strategy in strategies:
        derivation = strategy(expression, variable, point, direction, result)
        if derivation:
            return derivation
    return ()
