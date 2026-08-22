"""Family-level regressions that reject opaque exact-result fallbacks."""

from dataclasses import dataclass

import pytest

from stepsolver import ExactResult, Solver

_OPAQUE_RULE = "Compute exact result"
_LHOPITAL_RULES = (
    "Check direct substitution",
    "Apply L'Hôpital's rule",
    "Substitute into the transformed limit",
)
_REPEATED_LHOPITAL_RULES = (
    "Check direct substitution",
    "Apply L'Hôpital's rule",
    "Apply L'Hôpital's rule again",
    "Substitute into the transformed limit",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HumanMethodCase:
    """One representative of a family with its required classroom method."""

    family: str
    query: str
    expected_rules: tuple[str, ...]


def _case(
    family: str,
    query: str,
    *expected_rules: str,
) -> HumanMethodCase:
    return HumanMethodCase(
        family=family,
        query=query,
        expected_rules=expected_rules,
    )


_CASES = (
    _case("general powers", "diff(x^x,x)", "Use logarithmic differentiation"),
    _case("general powers", "diff((x+1)^x,x)", "Use logarithmic differentiation"),
    _case("general powers", "diff((x^2+1)^(3*x),x)", "Use logarithmic differentiation"),
    _case(
        "general powers",
        "diff((sin(x)+2)^cos(x),x)",
        "Use logarithmic differentiation",
    ),
    _case(
        "inverse functions by parts",
        "integrate(log(x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(log(2*x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(log(x+1),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(log(3*x-2),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(log(5-2*x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(atan(x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(atan(2*x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(atan(x+1),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(atan(3*x-2),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "inverse functions by parts",
        "integrate(atan(5-2*x),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    _case(
        "scaled inverse functions by parts",
        "integrate(-2*atan(3*x+4),x)",
        "Choose integration by parts",
        "Evaluate the simpler remaining integral",
    ),
    *(
        _case(
            "cyclic integration by parts",
            query,
            "Integrate by parts once",
            "Integrate by parts again and solve for the integral",
        )
        for query in (
            "integrate(exp(x)*sin(x),x)",
            "integrate(exp(2*x)*sin(3*x),x)",
            "integrate(2*exp(2*x)*cos(3*x),x)",
            "integrate(exp(-x)*cos(2*x),x)",
        )
    ),
    *(
        _case(
            "sine substitution",
            f"integrate(sin(x)^{power}*cos(x),x)",
            "Substitute the sine",
            "Apply the power rule and substitute back",
        )
        for power in range(1, 9)
    ),
    *(
        _case(
            "cosine substitution",
            f"integrate(sin(x)*cos(x)^{power},x)",
            "Substitute the cosine",
            "Apply the power rule and substitute back",
        )
        for power in range(2, 9)
    ),
    *(
        _case(
            "affine-angle trig substitution",
            query,
            substitution_rule,
            "Apply the power rule and substitute back",
        )
        for query, substitution_rule in (
            ("integrate(sin(2*x)^2*cos(2*x),x)", "Substitute the sine"),
            ("integrate(sin(3*x+1)^4*cos(3*x+1),x)", "Substitute the sine"),
            ("integrate(sin(2*x)*cos(2*x)^3,x)", "Substitute the cosine"),
            ("integrate(sin(4*x-1)*cos(4*x-1)^2,x)", "Substitute the cosine"),
            ("integrate(5*sin(2*x-1)^5*cos(2*x-1),x)", "Substitute the sine"),
            (
                "integrate(-3*sin(4*x+2)*cos(4*x+2)^4,x)",
                "Substitute the cosine",
            ),
        )
    ),
    *(
        _case(
            "inverse tangent substitution",
            query,
            "Substitute the repeated inner power",
            "Use the arctangent antiderivative",
        )
        for query in (
            "integrate((2*x+1)/(1+(x^2+x)^2),x)",
            "integrate(2*exp(2*x)/(1+exp(4*x)),x)",
            "integrate(cos(x)/(1+sin(x)^2),x)",
            "integrate(6*x^2/(1+x^6),x)",
            "integrate(3*cos(2*x)/(1+sin(2*x)^2),x)",
        )
    ),
    *(
        _case(
            "infinite telescoping sums",
            f"sum(1/(n*(n+{shift})),n,1,oo)",
            "Decompose the summand into partial fractions",
            "Cancel the telescoping terms",
        )
        for shift in range(1, 7)
    ),
    _case(
        "scaled telescoping sums",
        "sum(3/(n*(n+1)),n,1,oo)",
        "Decompose the summand into partial fractions",
        "Cancel the telescoping terms",
    ),
    _case(
        "shifted telescoping sums",
        "sum(1/(n*(n+3)),n,2,oo)",
        "Decompose the summand into partial fractions",
        "Cancel the telescoping terms",
    ),
    _case(
        "finite telescoping sums",
        "sum(2/(n*(n+2)),n,1,12)",
        "Decompose the summand into partial fractions",
        "Cancel the telescoping terms",
    ),
    *(
        _case(
            "weighted geometric sums",
            query,
            "Differentiate the geometric-series identity",
        )
        for query in (
            "sum(n/2^n,n,1,oo)",
            "sum(n/3^n,n,1,oo)",
            "sum(2*n*(1/2)^n,n,1,oo)",
            "sum(n*(-1/2)^n,n,1,oo)",
        )
    ),
    _case(
        "scaled exponential limits",
        "limit((exp(3*x)-1)/x,x,0)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "scaled exponential limits",
        "limit((exp(2*x)-1)/(3*x),x,0)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "scaled logarithm limits",
        "limit(log(1+2*x)/x,x,0)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "scaled logarithm limits",
        "limit(log(1+4*(x-1))/(2*(x-1)),x,1)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "scaled cosine limits",
        "limit((1-cos(4*x))/x^2,x,0)",
        *_REPEATED_LHOPITAL_RULES,
    ),
    _case(
        "scaled cosine limits",
        "limit((1-cos(3*(x+2)))/(2*(x+2)^2),x,-2)",
        *_REPEATED_LHOPITAL_RULES,
    ),
    _case(
        "shifted sine limits",
        "limit(sin(x-2)/(x-2),x,2)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "scaled shifted sine limits",
        "limit(sin(3*(x-2))/(2*(x-2)),x,2)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "shifted exponential limits",
        "limit((exp(x-2)-1)/(x-2),x,2)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "shifted logarithm limits",
        "limit(log(1+x-3)/(x-3),x,3)",
        *_LHOPITAL_RULES,
    ),
    _case(
        "shifted cosine limits",
        "limit((1-cos(x-4))/(x-4)^2,x,4)",
        *_REPEATED_LHOPITAL_RULES,
    ),
    _case(
        "exponential-definition limits",
        "limit((1+3/x)^x,x,oo)",
        "Use the exponential-definition limit",
    ),
    _case(
        "scaled exponential-definition limits",
        "limit((1+2/x)^(3*x),x,oo)",
        "Use the exponential-definition limit",
    ),
    _case(
        "negative exponential-definition increments",
        "limit((1-1/x)^x,x,oo)",
        "Use the exponential-definition limit",
    ),
    _case(
        "scaled negative exponential-definition increments",
        "limit((1-4/x)^(x/2),x,oo)",
        "Use the exponential-definition limit",
    ),
    _case(
        "variable-power limits",
        "limit(x^x,x,0,right)",
        "Rewrite the variable power exponentially",
        "Evaluate the exponent limit",
    ),
    *(
        _case(
            "radical rationalization",
            query,
            "Multiply by the conjugate",
            "Substitute into the rationalized expression",
        )
        for query in (
            "limit((sqrt(x+1)-1)/x,x,0)",
            "limit((sqrt(x+4)-2)/x,x,0)",
            "limit((sqrt(2*x+9)-3)/x,x,0)",
            "limit((sqrt(x+3)-2)/(x-1),x,1)",
            "limit((sqrt(2*x+7)-3)/(x-1),x,1)",
        )
    ),
    _case(
        "weighted geometric sums",
        "sum(k*(1/3)^k,k,1,oo)",
        "Differentiate the geometric-series identity",
    ),
    _case(
        "weighted geometric sums",
        "sum(4*k/5^k,k,1,oo)",
        "Differentiate the geometric-series identity",
    ),
)


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda case: f"{case.family}: {case.query}",
)
def test_human_method_family_never_degrades_to_an_exact_result_only(
    case: HumanMethodCase,
) -> None:
    """Nearby forms must retain their named, checkable classroom derivation."""
    result = Solver().solve(case.query)
    assert isinstance(result, ExactResult)
    actual_rules = tuple(step.rule for step in result.steps)
    assert actual_rules, (
        f"{case.family} produced an exact answer without worked steps for {case.query}"
    )
    assert _OPAQUE_RULE not in actual_rules, (
        f"{case.family} used the opaque fallback for {case.query}; "
        f"expected {case.expected_rules}"
    )
    assert actual_rules == case.expected_rules
