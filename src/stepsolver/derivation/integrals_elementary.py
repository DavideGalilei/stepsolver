"""Direct antiderivatives and linearity for elementary functions."""

from __future__ import annotations

import sympy as sp

from stepsolver.derivation.antiderivative_pairs import ELEMENTARY_PAIRS
from stepsolver.derivation.integrals_parts import derive_power_log_integral
from stepsolver.derivation.integrals_substitution import derive_linear_argument_integral
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendDerivative,
    BackendIdentity,
    BackendIntegral,
    BackendMathNote,
    BackendNotEqual,
    BackendSum,
)
from stepsolver.results import VerificationMethod

_MINIMUM_POLYNOMIAL_TERMS = 2


def _derive_power_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    if not integrand.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(integrand, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if sp.simplify(integrand - coefficient * variable**degree) != sp.Integer(0):
        return ()
    next_degree = degree + 1
    antiderivative = coefficient * variable**next_degree / next_degree
    formula = sp.Add(antiderivative, integration_constant, evaluate=False)
    if sp.simplify(formula - result) != sp.Integer(0):
        return ()
    pattern_variable = sp.Symbol("t", real=True)
    pattern_exponent = sp.Symbol("n", real=True)
    pattern_integrand = sp.Pow(pattern_variable, pattern_exponent, evaluate=False)
    pattern_result = sp.Add(
        sp.Mul(
            sp.Pow(pattern_variable, pattern_exponent + 1, evaluate=False),
            sp.Pow(pattern_exponent + 1, -1, evaluate=False),
            evaluate=False,
        ),
        integration_constant,
        evaluate=False,
    )
    return (
        BackendDerivationStep(
            rule="Use the power rule",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=formula,
            explanation=(
                "Increase the exponent by one, divide by the new exponent, and keep any "
                "constant coefficient."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the power-rule result recovers the original monomial."
            ),
            notes=(
                BackendMathNote(
                    label="General power rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=pattern_integrand,
                            variable=pattern_variable,
                        ),
                        right=pattern_result,
                    ),
                ),
                BackendMathNote(
                    label="Restriction",
                    expression=BackendNotEqual(
                        left=pattern_exponent,
                        right=sp.Integer(-1),
                    ),
                ),
            ),
        ),
    )


def derive_basic_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive direct elementary antiderivatives from differentiation rules."""
    power_log = derive_power_log_integral(integrand, variable, result)
    if power_log:
        return power_log
    integration_constant = sp.Symbol("C")
    direct_pairs = tuple(
        (pair.function(variable), pair.primitive(variable), f"Use the {pair.name} antiderivative")
        for pair in ELEMENTARY_PAIRS
    )
    candidates = (
        *direct_pairs,
        (sp.sinh(variable), sp.cosh(variable), "Use the hyperbolic sine antiderivative"),
        (sp.cosh(variable), sp.sinh(variable), "Use the hyperbolic cosine antiderivative"),
        (
            sp.tan(variable),
            -sp.log(sp.cos(variable)),
            "Use the logarithmic derivative of cosine",
        ),
        (
            sp.Pow(1 - variable**2, sp.Rational(-1, 2)),
            sp.asin(variable),
            "Use the inverse-sine derivative pattern",
        ),
        (
            sp.Pow(1 + variable**2, sp.Rational(-1, 2)),
            sp.asinh(variable),
            "Use the inverse-hyperbolic-sine derivative pattern",
        ),
        (
            variable * sp.Pow(variable**2 + 1, sp.Rational(-1, 2)),
            sp.sqrt(variable**2 + 1),
            "Use the reverse chain rule for the square root",
        ),
    )
    for candidate_integrand, antiderivative, rule in candidates:
        if sp.simplify(integrand - candidate_integrand) != sp.Integer(0):
            continue
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if sp.simplify(formula - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule=rule,
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=formula,
                explanation=(
                    "Use the matching derivative pair and include the integration constant."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "Differentiating the displayed antiderivative recovers the integrand."
                ),
                notes=(
                    BackendMathNote(
                        label="Derivative pair",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=candidate_integrand,
                        ),
                    ),
                ),
            ),
        )
    reverse_chain = derive_linear_argument_integral(
        integrand,
        variable,
        result,
        integration_constant,
    )
    if reverse_chain:
        return reverse_chain
    return _derive_power_antiderivative(
        integrand,
        variable,
        result,
        integration_constant,
    )


def derive_polynomial_sum_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a polynomial integral term by term using linearity."""
    if not integrand.is_polynomial(variable):
        return ()
    terms = tuple(integrand.as_ordered_terms())
    if len(terms) < _MINIMUM_POLYNOMIAL_TERMS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    integration_constant = sp.Symbol("C")
    formula = sp.Add(*antiderivatives, integration_constant, evaluate=False)
    if sp.simplify(formula - result) != sp.Integer(0):
        return ()
    split_integral = BackendSum(
        terms=tuple(BackendIntegral(integrand=term, variable=variable) for term in terms)
    )
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    term_notes = tuple(
        BackendMathNote(
            label=f"Integrate term {index}",
            expression=BackendIdentity(
                left=BackendIntegral(integrand=term, variable=variable),
                right=antiderivative,
            ),
        )
        for index, (term, antiderivative) in enumerate(
            zip(terms, antiderivatives, strict=True),
            start=1,
        )
    )
    return (
        BackendDerivationStep(
            rule="Split the integral across the sum",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=split_integral,
            explanation="Use linearity to integrate each polynomial term separately.",
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Adding the separated integrands recovers the original polynomial.",
            notes=(
                BackendMathNote(
                    label="Linearity rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=sp.Add(function_f, function_g, evaluate=False),
                            variable=variable,
                        ),
                        right=BackendSum(
                            terms=(
                                BackendIntegral(integrand=function_f, variable=variable),
                                BackendIntegral(integrand=function_g, variable=variable),
                            )
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate each term",
            before=split_integral,
            after=formula,
            explanation="Apply the power rule to each term and add one integration constant.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the combined result recovers the polynomial.",
            notes=term_notes,
        ),
    )


def derive_constant_multiple_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive direct elementary integrals with a constant multiplier."""
    variable_factors = tuple(
        factor for factor in integrand.as_ordered_factors() if factor.has(variable)
    )
    if len(variable_factors) != 1:
        return ()
    variable_part = variable_factors[0]
    coefficient = sp.simplify(integrand / variable_part)
    if coefficient == sp.Integer(1) or coefficient.has(variable):
        return ()
    for pair in ELEMENTARY_PAIRS:
        candidate = pair.function(variable)
        antiderivative = pair.primitive(variable)
        if sp.simplify(variable_part - candidate) != sp.Integer(0):
            continue
        integration_constant = sp.Symbol("C")
        formula = sp.Add(
            sp.Mul(coefficient, antiderivative),
            integration_constant,
            evaluate=False,
        )
        if sp.simplify(formula - result) != sp.Integer(0):
            return ()
        reduced_integral = BackendIntegral(
            integrand=variable_part,
            variable=variable,
            coefficient=coefficient,
        )
        return (
            BackendDerivationStep(
                rule="Factor out the constant",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=reduced_integral,
                explanation="Move the constant multiplier outside the integral.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Multiplying the reduced integrand by the constant recovers the original."
                ),
            ),
            BackendDerivationStep(
                rule=f"Use the {pair.name} antiderivative",
                before=reduced_integral,
                after=formula,
                explanation="Apply the basic derivative pair, preserving the outside constant.",
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the result recovers the original integrand.",
                notes=(
                    BackendMathNote(
                        label="Derivative pair",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=candidate,
                        ),
                    ),
                ),
            ),
        )
    return ()
