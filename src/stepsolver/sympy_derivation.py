"""Backend-local derivation strategies for human-readable solution steps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import sympy as sp

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
class BackendEvaluationAtBounds:
    """An antiderivative evaluated between lower and upper bounds."""

    expression: sp.Basic
    variable: sp.Symbol
    lower: sp.Basic
    upper: sp.Basic


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


type BackendExpression = (
    EquationBackendExpression
    | BackendIntegral
    | BackendDifferential
    | BackendDerivative
    | BackendIntegrationByPartsRule
    | BackendQuadraticSolutions
    | BackendEvaluationAtBounds
    | BackendLimit
    | BackendNotEqual
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


_LINEAR_DEGREE = 1
_QUADRATIC_DEGREE = 2
_CONSTANT_DEGREE = 0
_BINARY_ARITY = 2
_MINIMUM_PRODUCT_FACTORS = 2


def _is_basic_sequence(value: object) -> TypeGuard[Sequence[sp.Basic]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    items = cast("Sequence[object]", value)
    return all(isinstance(item, sp.Basic) for item in items)


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


def _equivalent_step(
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
) -> BackendDerivationStep:
    if _solution_set(before, variable) != _solution_set(after, variable):
        message = "a proposed derivation step changed the equation's solution set"
        raise ValueError(message)
    return BackendDerivationStep(
        rule=rule,
        before=before,
        after=after,
        explanation=explanation,
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail="Both forms have the same solution set for the target variable.",
    )


def _solution_set(
    expression: EquationBackendExpression,
    variable: sp.Symbol,
) -> frozenset[str]:
    equations = expression if isinstance(expression, tuple) else (expression,)
    roots: set[str] = set()
    for equation in equations:
        solved = sp.solve(equation, variable)
        if not _is_basic_sequence(solved):
            message = "equation verification did not produce a root sequence"
            raise TypeError(message)
        roots.update(str(sp.simplify(root)) for root in solved)
    return frozenset(roots)


def _relations(variable: sp.Symbol, roots: tuple[sp.Basic, ...]) -> tuple[sp.Basic, ...]:
    return tuple(sp.Eq(variable, root) for root in roots)


def _append_if_changed(
    steps: list[BackendDerivationStep],
    *,
    rule: str,
    before: EquationBackendExpression,
    after: EquationBackendExpression,
    explanation: str,
    variable: sp.Symbol,
) -> EquationBackendExpression:
    if str(before) == str(after):
        return before
    steps.append(
        _equivalent_step(
            rule=rule,
            before=before,
            after=after,
            explanation=explanation,
            variable=variable,
        )
    )
    return after


def _linear_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    coefficient = polynomial.coeff_monomial(variable)
    constant = polynomial.coeff_monomial(1)
    isolated_term = sp.Eq(coefficient * variable, -constant)
    current: EquationBackendExpression = equation
    current = _append_if_changed(
        steps,
        rule="Collect variable terms",
        before=current,
        after=isolated_term,
        explanation=(
            "Move every variable term to one side and every constant term to the other side."
        ),
        variable=variable,
    )
    final: EquationBackendExpression = (
        sp.Eq(variable, roots[0]) if len(roots) == _LINEAR_DEGREE else _relations(variable, roots)
    )
    _append_if_changed(
        steps,
        rule="Divide by the coefficient",
        before=current,
        after=final,
        explanation="Divide both sides by the coefficient of the variable.",
        variable=variable,
    )
    return tuple(steps)


def _quadratic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    steps: list[BackendDerivationStep] = []
    expression = polynomial.as_expr()
    factored = sp.factor(expression)
    current: EquationBackendExpression = equation
    if str(factored) != str(expression) and (factored.is_Mul or factored.is_Pow):
        _, factor_pairs = sp.factor_list(expression, variable)
        factor_bases = tuple(
            factor for factor, _multiplicity in factor_pairs if factor.has(variable)
        )
        has_nonlinear_factor = any(
            sp.Poly(factor, variable).degree() != 1 for factor in factor_bases
        )
        if not factor_bases or has_nonlinear_factor:
            return ()
        factored_equation = sp.Eq(factored, 0)
        current = _append_if_changed(
            steps,
            rule="Factor the quadratic",
            before=current,
            after=factored_equation,
            explanation="Rewrite the quadratic as a product of linear factors.",
            variable=variable,
        )
        factor_equations = tuple(sp.Eq(factor, 0) for factor in factor_bases)
        separated_factors: EquationBackendExpression = (
            factor_equations[0] if len(factor_equations) == 1 else factor_equations
        )
        current = _append_if_changed(
            steps,
            rule=(
                "Set the repeated factor equal to zero"
                if len(factor_equations) == 1
                else "Apply the zero-product property"
            ),
            before=current,
            after=separated_factors,
            explanation=(
                "A power is zero only when its base is zero."
                if len(factor_equations) == 1
                else "A product is zero only when at least one of its factors is zero."
            ),
            variable=variable,
        )
        factor_roots = tuple(
            -sp.Poly(factor, variable).coeff_monomial(1)
            / sp.Poly(factor, variable).coeff_monomial(variable)
            for factor in factor_bases
        )
        _append_if_changed(
            steps,
            rule="Solve each factor",
            before=current,
            after=_relations(variable, factor_roots),
            explanation="Solve each resulting linear equation and combine the solutions.",
            variable=variable,
        )
        return tuple(steps)

    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    discriminant = sp.expand(coefficient_b**2 - 4 * coefficient_a * coefficient_c)
    if not roots and discriminant.is_negative is True:
        discriminant_symbol = sp.Symbol("delta", real=True)
        discriminant_value = sp.Eq(discriminant_symbol, discriminant, evaluate=False)
        generic_discriminant = sp.Eq(
            discriminant_symbol,
            sp.Symbol("b") ** 2 - 4 * sp.Symbol("a") * sp.Symbol("c"),
            evaluate=False,
        )
        return (
            BackendDerivationStep(
                rule="Calculate the discriminant",
                before=equation,
                after=discriminant_value,
                explanation="Compute the discriminant to determine the type of roots.",
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="The discriminant was evaluated from the exact coefficients.",
                notes=(
                    BackendMathNote(
                        label="Discriminant rule",
                        expression=generic_discriminant,
                    ),
                    BackendMathNote(
                        label="Coefficients",
                        expression=(
                            sp.Eq(sp.Symbol("a"), coefficient_a, evaluate=False),
                            sp.Eq(sp.Symbol("b"), coefficient_b, evaluate=False),
                            sp.Eq(sp.Symbol("c"), coefficient_c, evaluate=False),
                        ),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Conclude there are no real solutions",
                before=discriminant_value,
                after=(),
                explanation=(
                    "A negative discriminant means the quadratic has no roots in the real "
                    "number system."
                ),
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="Solving over the real numbers produces the empty set.",
            ),
        )
    denominator = 2 * coefficient_a
    negative_numerator = sp.Add(-coefficient_b, -sp.sqrt(discriminant), evaluate=False)
    positive_numerator = sp.Add(-coefficient_b, sp.sqrt(discriminant), evaluate=False)
    generic_a = sp.Symbol("a", nonzero=True)
    generic_b = sp.Symbol("b")
    generic_c = sp.Symbol("c")
    generic_formula_discriminant = generic_b**2 - 4 * generic_a * generic_c
    steps.append(
        BackendDerivationStep(
            rule="Apply the quadratic formula",
            before=current,
            after=BackendQuadraticSolutions(
                variable=variable,
                negative_numerator=negative_numerator,
                positive_numerator=positive_numerator,
                denominator=denominator,
            ),
            explanation=(
                "Identify a, b, and c, calculate the discriminant, then substitute into the "
                "quadratic formula."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The displayed quadratic-formula roots match the exact roots.",
            notes=(
                BackendMathNote(
                    label="Quadratic formula",
                    expression=BackendQuadraticSolutions(
                        variable=variable,
                        negative_numerator=sp.Add(
                            -generic_b,
                            -sp.sqrt(generic_formula_discriminant),
                            evaluate=False,
                        ),
                        positive_numerator=sp.Add(
                            -generic_b,
                            sp.sqrt(generic_formula_discriminant),
                            evaluate=False,
                        ),
                        denominator=2 * generic_a,
                    ),
                ),
                BackendMathNote(
                    label="Coefficient a",
                    expression=BackendIdentity(left=generic_a, right=coefficient_a),
                ),
                BackendMathNote(
                    label="Coefficient b",
                    expression=BackendIdentity(left=generic_b, right=coefficient_b),
                ),
                BackendMathNote(
                    label="Coefficient c",
                    expression=BackendIdentity(left=generic_c, right=coefficient_c),
                ),
                BackendMathNote(
                    label="Discriminant",
                    expression=BackendIdentity(
                        left=sp.Symbol("delta"),
                        right=discriminant,
                    ),
                ),
            ),
        )
    )
    return tuple(steps)


def derive_polynomial_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive detailed steps for rational, linear, and quadratic equations."""
    steps: list[BackendDerivationStep] = []
    difference = sp.together(equation.lhs - equation.rhs)
    numerator, denominator = sp.fraction(difference)
    expanded = sp.expand(numerator)
    normalized = sp.Eq(expanded, 0, evaluate=False)
    current: EquationBackendExpression = equation
    if str(denominator) != "1":
        current = _append_if_changed(
            steps,
            rule="Clear the denominators",
            before=current,
            after=normalized,
            explanation=("Multiply through by the common denominator, which must be nonzero."),
            variable=variable,
        )
    polynomial = sp.Poly(expanded, variable)
    degree = polynomial.degree()
    if degree == _CONSTANT_DEGREE:
        current = _append_if_changed(
            steps,
            rule="Simplify the equation",
            before=current,
            after=normalized,
            explanation="Combine like terms on both sides.",
            variable=variable,
        )
        steps.append(
            BackendDerivationStep(
                rule="Conclude there are no solutions",
                before=current,
                after=(),
                explanation="A nonzero constant cannot equal zero.",
                verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
                verification_detail="The simplified contradiction has an empty solution set.",
            )
        )
        return tuple(steps)
    if degree == _LINEAR_DEGREE:
        detail = _linear_steps(normalized if steps else equation, variable, polynomial, roots)
    elif degree == _QUADRATIC_DEGREE:
        normalized_current = normalized
        if not steps:
            if sp.simplify(equation.rhs) == sp.Integer(0):
                normalized_current = equation
            else:
                current = _append_if_changed(
                    steps,
                    rule="Write in standard form",
                    before=current,
                    after=normalized,
                    explanation="Move every term to one side and combine like terms.",
                    variable=variable,
                )
                if isinstance(current, sp.Equality):
                    normalized_current = current
        detail = _quadratic_steps(normalized_current, variable, polynomial, roots)
    else:
        return ()
    steps.extend(detail)
    return tuple(steps)


def _verified_derivative_steps(
    *,
    rule: str,
    expression: sp.Basic,
    variable: sp.Symbol,
    raw_derivative: sp.Basic,
    result: sp.Basic,
    explanation: str,
    notes: tuple[BackendMathNote, ...],
    show_simplification: bool = False,
) -> tuple[BackendDerivationStep, ...]:
    if str(sp.simplify(raw_derivative - result)) != "0":
        message = "the derivative rule did not match the exact backend result"
        raise ValueError(message)
    steps = [
        BackendDerivationStep(
            rule=rule,
            before=BackendDerivative(expression=expression, variable=variable),
            after=raw_derivative,
            explanation=explanation,
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "The displayed rule was applied symbolically and checked against the exact "
                "derivative."
            ),
            notes=notes,
        )
    ]
    if show_simplification and str(raw_derivative) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Simplify the derivative",
                before=raw_derivative,
                after=result,
                explanation="Combine factors and like terms into a cleaner final form.",
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between both derivative forms gives zero."
                ),
            )
        )
    return tuple(steps)


def _generic_product_rule(variable: sp.Symbol) -> BackendIdentity:
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    return BackendIdentity(
        left=BackendDerivative(
            expression=sp.Mul(function_f, function_g, evaluate=False),
            variable=variable,
        ),
        right=BackendSum(
            terms=(
                BackendProduct(
                    factors=(
                        BackendDerivative(expression=function_f, variable=variable),
                        function_g,
                    )
                ),
                BackendProduct(
                    factors=(
                        function_f,
                        BackendDerivative(expression=function_g, variable=variable),
                    )
                ),
            )
        ),
    )


def _generic_quotient_rule(variable: sp.Symbol) -> BackendIdentity:
    function_f = sp.Function("f")(variable)
    function_g = sp.Function("g")(variable)
    numerator = BackendDifference(
        left=BackendProduct(
            factors=(
                BackendDerivative(expression=function_f, variable=variable),
                function_g,
            )
        ),
        right=BackendProduct(
            factors=(
                function_f,
                BackendDerivative(expression=function_g, variable=variable),
            )
        ),
    )
    return BackendIdentity(
        left=BackendDerivative(
            expression=sp.Mul(function_f, sp.Pow(function_g, -1, evaluate=False), evaluate=False),
            variable=variable,
        ),
        right=BackendQuotient(
            numerator=numerator,
            denominator=sp.Pow(function_g, 2, evaluate=False),
        ),
    )


def _derive_quotient(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    numerator, denominator = sp.fraction(expression)
    if denominator == sp.Integer(1) or not denominator.has(variable):
        return ()
    numerator_derivative = sp.diff(numerator, variable)
    denominator_derivative = sp.diff(denominator, variable)
    first_term = sp.Mul(numerator_derivative, denominator)
    second_term = sp.Mul(numerator, denominator_derivative)
    raw_numerator = sp.Add(
        first_term,
        -second_term,
        evaluate=False,
    )
    raw_derivative = sp.Mul(
        raw_numerator,
        sp.Pow(denominator, -2, evaluate=False),
        evaluate=False,
    )
    return _verified_derivative_steps(
        rule="Apply the quotient rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Differentiate the numerator and denominator separately, then apply the quotient "
            "rule in its standard order."
        ),
        notes=(
            BackendMathNote(label="Quotient rule", expression=_generic_quotient_rule(variable)),
            BackendMathNote(
                label="Numerator derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=numerator, variable=variable),
                    right=numerator_derivative,
                ),
            ),
            BackendMathNote(
                label="Denominator derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=denominator, variable=variable),
                    right=denominator_derivative,
                ),
            ),
        ),
        show_simplification=True,
    )


def _derive_product(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    variable_factors = tuple(
        factor for factor in expression.as_ordered_factors() if factor.has(variable)
    )
    if len(variable_factors) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    first_factor = variable_factors[0]
    second_factor = sp.Mul(*variable_factors[1:])
    constant = sp.simplify(expression / (first_factor * second_factor))
    product_derivative = sp.Add(
        sp.Mul(sp.diff(first_factor, variable), second_factor, evaluate=False),
        sp.Mul(first_factor, sp.diff(second_factor, variable), evaluate=False),
        evaluate=False,
    )
    raw_derivative = (
        product_derivative
        if constant == sp.Integer(1)
        else sp.Mul(constant, product_derivative, evaluate=False)
    )
    return _verified_derivative_steps(
        rule="Apply the product rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation="Differentiate one factor at a time while leaving the other unchanged.",
        notes=(
            BackendMathNote(label="Product rule", expression=_generic_product_rule(variable)),
            BackendMathNote(
                label="First factor derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=first_factor, variable=variable),
                    right=sp.diff(first_factor, variable),
                ),
            ),
            BackendMathNote(
                label="Second factor derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=second_factor, variable=variable),
                    right=sp.diff(second_factor, variable),
                ),
            ),
        ),
        show_simplification=True,
    )


def _derive_sum(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    terms = tuple(expression.as_ordered_terms())
    if len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    derivatives = tuple(sp.diff(term, variable) for term in terms)
    raw_derivative = sp.Add(*derivatives, evaluate=False)
    generic_f = sp.Function("f")(variable)
    generic_g = sp.Function("g")(variable)
    return _verified_derivative_steps(
        rule="Differentiate term by term",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation="Use linearity, then differentiate each term with its matching rule.",
        notes=(
            BackendMathNote(
                label="Sum rule",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=sp.Add(generic_f, generic_g, evaluate=False),
                        variable=variable,
                    ),
                    right=BackendSum(
                        terms=(
                            BackendDerivative(expression=generic_f, variable=variable),
                            BackendDerivative(expression=generic_g, variable=variable),
                        )
                    ),
                ),
            ),
            *tuple(
                BackendMathNote(
                    label=f"Term {index}",
                    expression=BackendIdentity(
                        left=BackendDerivative(expression=term, variable=variable),
                        right=derivative,
                    ),
                )
                for index, (term, derivative) in enumerate(
                    zip(terms, derivatives, strict=True),
                    start=1,
                )
            ),
        ),
    )


def _derive_monomial(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not expression.is_polynomial(variable):
        return ()
    polynomial = sp.Poly(expression, variable)
    degree = polynomial.degree()
    coefficient = polynomial.coeff_monomial(variable**degree)
    if str(sp.simplify(expression - coefficient * variable**degree)) != "0":
        return ()
    raw_derivative = coefficient * degree * variable ** (degree - 1)
    pattern_variable = sp.Symbol("t", real=True)
    pattern_exponent = sp.Symbol("n", real=True)
    return _verified_derivative_steps(
        rule="Use the power rule",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Multiply by the exponent, then decrease the exponent by one; preserve the "
            "constant coefficient."
        ),
        notes=(
            BackendMathNote(
                label="General power rule",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=sp.Pow(pattern_variable, pattern_exponent, evaluate=False),
                        variable=pattern_variable,
                    ),
                    right=sp.Mul(
                        pattern_exponent,
                        sp.Pow(pattern_variable, pattern_exponent - 1, evaluate=False),
                        evaluate=False,
                    ),
                ),
            ),
        ),
    )


def _negative_sine(argument: sp.Basic) -> sp.Basic:
    return -sp.sin(argument)


def _reciprocal(argument: sp.Basic) -> sp.Basic:
    return sp.Pow(argument, -1, evaluate=False)


def _derive_function_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    candidates: tuple[
        tuple[Callable[[sp.Basic], sp.Basic], Callable[[sp.Basic], sp.Basic], str],
        ...,
    ] = (
        (sp.sin, sp.cos, "Differentiate the sine"),
        (sp.cos, _negative_sine, "Differentiate the cosine"),
        (sp.exp, sp.exp, "Differentiate the exponential"),
        (sp.log, _reciprocal, "Differentiate the logarithm"),
    )
    for function, outer_derivative, direct_rule in candidates:
        if expression.func != function or len(expression.args) != 1:
            continue
        argument = expression.args[0]
        inner_derivative = sp.diff(argument, variable)
        outer_value = outer_derivative(argument)
        raw_derivative = (
            outer_value
            if inner_derivative == sp.Integer(1)
            else sp.Mul(outer_value, inner_derivative, evaluate=False)
        )
        is_direct = argument == variable
        notes: tuple[BackendMathNote, ...] = (
            BackendMathNote(
                label="Outer derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(
                        expression=function(sp.Symbol("u", real=True)),
                        variable=sp.Symbol("u", real=True),
                    ),
                    right=outer_derivative(sp.Symbol("u", real=True)),
                ),
            ),
        )
        if not is_direct:
            notes = (
                *notes,
                BackendMathNote(
                    label="Inner derivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(expression=argument, variable=variable),
                        right=inner_derivative,
                    ),
                ),
            )
        return _verified_derivative_steps(
            rule=direct_rule if is_direct else "Apply the chain rule",
            expression=expression,
            variable=variable,
            raw_derivative=raw_derivative,
            result=result,
            explanation=(
                "Use the basic derivative pair."
                if is_direct
                else "Differentiate the outer function, then multiply by the inner derivative."
            ),
            notes=notes,
        )
    return ()


def _derive_power_chain(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    if not expression.is_Pow or len(expression.args) != _BINARY_ARITY:
        return ()
    base, exponent = expression.args
    if exponent.has(variable):
        return ()
    base_derivative = sp.diff(base, variable)
    outer_power_derivative = sp.Mul(
        exponent,
        sp.Pow(base, exponent - 1, evaluate=False),
        evaluate=False,
    )
    raw_derivative = (
        outer_power_derivative
        if base_derivative == sp.Integer(1)
        else sp.Mul(outer_power_derivative, base_derivative, evaluate=False)
    )
    return _verified_derivative_steps(
        rule="Apply the power and chain rules",
        expression=expression,
        variable=variable,
        raw_derivative=raw_derivative,
        result=result,
        explanation=(
            "Differentiate the outer power, then multiply by the derivative of its base."
        ),
        notes=(
            BackendMathNote(
                label="Base derivative",
                expression=BackendIdentity(
                    left=BackendDerivative(expression=base, variable=variable),
                    right=base_derivative,
                ),
            ),
        ),
    )


def derive_derivative(
    expression: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a first derivative using the most specific familiar rule."""
    strategies = (
        _derive_quotient,
        _derive_monomial,
        _derive_sum,
        _derive_product,
        _derive_power_chain,
        _derive_function_chain,
    )
    for strategy in strategies:
        derivation = strategy(expression, variable, result)
        if derivation:
            return derivation
    return ()


def _completion_notes(
    *,
    denominator: sp.Basic,
    variable: sp.Symbol,
    coefficient_a: sp.Basic,
    coefficient_b: sp.Basic,
    coefficient_c: sp.Basic,
    radius_squared: sp.Basic,
    completed_denominator: sp.Basic,
) -> tuple[BackendMathNote, ...]:
    """Build a concrete, student-first completing-the-square explanation."""
    normalized_linear_coefficient = sp.simplify(coefficient_b / coefficient_a)
    normalized_constant = sp.simplify(coefficient_c / coefficient_a)
    half_linear_coefficient = sp.simplify(normalized_linear_coefficient / 2)
    square_completion = sp.simplify(half_linear_coefficient**2)
    monic_denominator = sp.Add(
        variable**2,
        sp.Mul(normalized_linear_coefficient, variable, evaluate=False),
        normalized_constant,
        evaluate=False,
    )
    expanded_completion = sp.Add(
        variable**2,
        sp.Mul(normalized_linear_coefficient, variable, evaluate=False),
        square_completion,
        radius_squared,
        evaluate=False,
    )
    pattern_variable = sp.Symbol("z", real=True)
    pattern_linear = sp.Symbol("p", real=True)
    pattern_constant = sp.Symbol("q", real=True)
    generic_quadratic = sp.Add(
        pattern_variable**2,
        sp.Mul(pattern_linear, pattern_variable, evaluate=False),
        pattern_constant,
        evaluate=False,
    )
    generic_completed_quadratic = sp.Add(
        sp.Pow(
            sp.Add(pattern_variable, pattern_linear / 2, evaluate=False),
            2,
            evaluate=False,
        ),
        sp.Add(pattern_constant, -(pattern_linear**2 / 4), evaluate=False),
        evaluate=False,
    )
    notes: list[BackendMathNote] = []
    if str(coefficient_a) != "1":
        notes.append(
            BackendMathNote(
                label="First factor the leading coefficient",
                expression=BackendIdentity(
                    left=denominator,
                    right=sp.Mul(coefficient_a, monic_denominator, evaluate=False),
                ),
            )
        )
    notes.extend(
        (
            BackendMathNote(
                label="Take half the linear coefficient, then square it",
                expression=BackendIdentity(
                    left=sp.Pow(half_linear_coefficient, 2, evaluate=False),
                    right=square_completion,
                ),
            ),
            BackendMathNote(
                label="Add and subtract that number",
                expression=BackendIdentity(left=monic_denominator, right=expanded_completion),
            ),
            BackendMathNote(
                label="Recognize the perfect square",
                expression=BackendIdentity(left=monic_denominator, right=completed_denominator),
            ),
            BackendMathNote(
                label="General pattern",
                expression=BackendIdentity(
                    left=generic_quadratic,
                    right=generic_completed_quadratic,
                ),
            ),
        )
    )
    return tuple(notes)


def _build_reciprocal_quadratic_steps(
    *,
    integrand: sp.Basic,
    variable: sp.Symbol,
    denominator: sp.Basic,
    coefficient_a: sp.Basic,
    coefficient_b: sp.Basic,
    coefficient_c: sp.Basic,
    radius_squared: sp.Basic,
    completed_denominator: sp.Basic,
    completed_integrand: sp.Basic,
    shift: sp.Basic,
    radius: sp.Basic,
    substitution_variable: sp.Symbol,
    unit_integrand: sp.Basic,
    normalized_argument: sp.Basic,
    normalized_coefficient: sp.Basic,
    coefficient_is_one: bool,
    formula_in_substitution_variable: sp.Basic,
    formula: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    """Choose only the transformations that add pedagogical value."""
    displayed_coefficient = None if coefficient_is_one else normalized_coefficient
    original_integral = BackendIntegral(integrand=integrand, variable=variable)
    completed_integral = BackendIntegral(integrand=completed_integrand, variable=variable)
    normalized_integral = BackendIntegral(
        integrand=unit_integrand,
        variable=substitution_variable,
        coefficient=displayed_coefficient,
    )
    needs_completion = coefficient_b != sp.Integer(0)
    needs_substitution = normalized_argument != variable or not coefficient_is_one
    steps: list[BackendDerivationStep] = []
    if needs_completion:
        steps.append(
            BackendDerivationStep(
                rule="Complete the square",
                before=original_integral,
                after=completed_integral,
                explanation=(
                    "Rewrite the quadratic denominator as a shifted square plus a positive "
                    "constant so it can match a standard integration rule."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail=(
                    "Simplifying the difference between the two integrands gives zero."
                ),
                notes=_completion_notes(
                    denominator=denominator,
                    variable=variable,
                    coefficient_a=coefficient_a,
                    coefficient_b=coefficient_b,
                    coefficient_c=coefficient_c,
                    radius_squared=radius_squared,
                    completed_denominator=completed_denominator,
                ),
            )
        )
    if needs_substitution:
        scaled_substitution_variable = (
            substitution_variable
            if radius == sp.Integer(1)
            else sp.Mul(radius, substitution_variable, evaluate=False)
        )
        steps.append(
            BackendDerivationStep(
                rule="Substitute to get a unit denominator",
                before=completed_integral if needs_completion else original_integral,
                after=normalized_integral,
                explanation=(
                    "Scale the shifted variable so the denominator becomes one plus its square. "
                    "Transform the differential at the same time."
                ),
                verification_method=VerificationMethod.SUBSTITUTION,
                verification_detail=(
                    "Replacing the new variable and its differential recovers the previous "
                    "integral."
                ),
                notes=(
                    BackendMathNote(
                        label="Choose the substitution",
                        expression=sp.Eq(
                            substitution_variable,
                            normalized_argument,
                            evaluate=False,
                        ),
                    ),
                    BackendMathNote(
                        label="Rewrite the shifted term",
                        expression=BackendIdentity(
                            left=shift,
                            right=scaled_substitution_variable,
                        ),
                    ),
                    BackendMathNote(
                        label="Change the differential",
                        expression=BackendIdentity(
                            left=BackendDifferential(variable=variable),
                            right=BackendDifferential(
                                variable=substitution_variable,
                                coefficient=None if radius == sp.Integer(1) else radius,
                            ),
                        ),
                    ),
                ),
            )
        )
    steps.append(
        BackendDerivationStep(
            rule="Use the basic arctangent rule",
            before=normalized_integral if needs_substitution else original_integral,
            after=formula_in_substitution_variable if needs_substitution else formula,
            explanation=(
                "The remaining integral is the derivative pattern for the arctangent function."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the arctangent expression recovers the displayed integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=unit_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.Add(
                            sp.atan(substitution_variable),
                            integration_constant,
                            evaluate=False,
                        ),
                    ),
                ),
            ),
        )
    )
    if needs_substitution:
        steps.append(
            BackendDerivationStep(
                rule="Substitute back",
                before=formula_in_substitution_variable,
                after=formula,
                explanation=(
                    "Replace the temporary variable with its expression in the original variable."
                ),
                verification_method=VerificationMethod.SUBSTITUTION,
                verification_detail=(
                    "Direct substitution gives an antiderivative equivalent to the exact result."
                ),
                notes=(
                    BackendMathNote(
                        label="Replace the temporary variable",
                        expression=sp.Eq(
                            substitution_variable,
                            normalized_argument,
                            evaluate=False,
                        ),
                    ),
                ),
            )
        )
    return tuple(steps)


def derive_log_derivative_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a logarithmic antiderivative using denominator substitution."""
    numerator, denominator = sp.fraction(sp.together(integrand))
    denominator_derivative = sp.diff(denominator, variable)
    if denominator_derivative == sp.Integer(0):
        return ()
    coefficient = sp.simplify(numerator / denominator_derivative)
    denominator_is_positive = denominator.is_positive is True
    if not denominator_is_positive and denominator.is_polynomial(variable):
        polynomial = sp.Poly(denominator, variable)
        if polynomial.degree() == _QUADRATIC_DEGREE:
            leading = polynomial.coeff_monomial(variable**2)
            linear = polynomial.coeff_monomial(variable)
            constant = polynomial.coeff_monomial(1)
            discriminant = sp.simplify(linear**2 - 4 * leading * constant)
            denominator_is_positive = (
                leading.is_positive is True and discriminant.is_negative is True
            )
    if coefficient.has(variable) or not denominator_is_positive:
        return ()
    substitution_variable = sp.Symbol("u", positive=True)
    unit_integrand = sp.Pow(substitution_variable, -1, evaluate=False)
    displayed_coefficient = None if coefficient == sp.Integer(1) else coefficient
    integration_constant = sp.Symbol("C")
    logarithm_in_substitution_variable = sp.log(substitution_variable)
    logarithm_in_original_variable = sp.log(denominator)
    formula_in_substitution_variable_term = (
        logarithm_in_substitution_variable
        if displayed_coefficient is None
        else sp.Mul(coefficient, logarithm_in_substitution_variable, evaluate=False)
    )
    formula_term = (
        logarithm_in_original_variable
        if displayed_coefficient is None
        else sp.Mul(coefficient, logarithm_in_original_variable, evaluate=False)
    )
    formula_in_substitution_variable = sp.Add(
        formula_in_substitution_variable_term,
        integration_constant,
        evaluate=False,
    )
    formula = sp.Add(formula_term, integration_constant, evaluate=False)
    if str(sp.simplify(sp.diff(formula, variable) - integrand)) != "0":
        message = "the logarithmic substitution failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the logarithmic antiderivative differs from the exact result"
        raise ValueError(message)
    transformed_integral = BackendIntegral(
        integrand=unit_integrand,
        variable=substitution_variable,
        coefficient=displayed_coefficient,
    )
    return (
        BackendDerivationStep(
            rule="Substitute the denominator",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "The numerator is a constant multiple of the denominator's derivative, so use "
                "the denominator as the new variable."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the denominator and its differential recovers the original integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=sp.Eq(
                        substitution_variable,
                        denominator,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Differentiate the substitution",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution_variable),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=denominator_derivative,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the logarithm rule",
            before=transformed_integral,
            after=formula_in_substitution_variable,
            explanation="The transformed integrand is the reciprocal function.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the logarithm recovers the reciprocal integrand."
            ),
            notes=(
                BackendMathNote(
                    label="The substitution is positive",
                    expression=sp.Gt(denominator, sp.Integer(0), evaluate=False),
                ),
                BackendMathNote(
                    label="Rule for positive inputs",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=unit_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.Add(
                            logarithm_in_substitution_variable,
                            integration_constant,
                            evaluate=False,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=formula_in_substitution_variable,
            after=formula,
            explanation="Replace the temporary variable with the original denominator.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Direct substitution gives an antiderivative equivalent to the exact result."
            ),
            notes=(
                BackendMathNote(
                    label="Replace the temporary variable",
                    expression=sp.Eq(
                        substitution_variable,
                        denominator,
                        evaluate=False,
                    ),
                ),
            ),
        ),
    )


def _negative_cosine(argument: sp.Basic) -> sp.Basic:
    return -sp.cos(argument)


def _derive_reverse_chain_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
    integration_constant: sp.Symbol,
) -> tuple[BackendDerivationStep, ...]:
    candidates: tuple[
        tuple[Callable[[sp.Basic], sp.Basic], Callable[[sp.Basic], sp.Basic], str],
        ...,
    ] = (
        (sp.sin, _negative_cosine, "Apply the reverse chain rule for sine"),
        (sp.cos, sp.sin, "Apply the reverse chain rule for cosine"),
        (sp.exp, sp.exp, "Apply the reverse chain rule for the exponential"),
    )
    for function, outer_antiderivative, rule in candidates:
        if integrand.func != function or len(integrand.args) != 1:
            continue
        argument = integrand.args[0]
        inner_derivative = sp.simplify(sp.diff(argument, variable))
        if inner_derivative.has(variable) or inner_derivative == sp.Integer(0):
            return ()
        antiderivative = outer_antiderivative(argument) / inner_derivative
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if str(sp.simplify(formula - result)) != "0":
            return ()
        inner_variable = sp.Symbol("u", real=True)
        return (
            BackendDerivationStep(
                rule=rule,
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=formula,
                explanation=(
                    "Divide the outer antiderivative by the constant derivative of the inner "
                    "function."
                ),
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail=(
                    "The chain rule confirms that differentiating the result recovers the "
                    "integrand."
                ),
                notes=(
                    BackendMathNote(
                        label="Inner function",
                        expression=sp.Eq(inner_variable, argument, evaluate=False),
                    ),
                    BackendMathNote(
                        label="Inner derivative",
                        expression=BackendIdentity(
                            left=BackendDerivative(expression=argument, variable=variable),
                            right=inner_derivative,
                        ),
                    ),
                    BackendMathNote(
                        label="Check with the chain rule",
                        expression=BackendIdentity(
                            left=BackendDerivative(
                                expression=antiderivative,
                                variable=variable,
                            ),
                            right=integrand,
                        ),
                    ),
                ),
            ),
        )
    return ()


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
    if str(sp.simplify(integrand - coefficient * variable**degree)) != "0":
        return ()
    next_degree = degree + 1
    antiderivative = coefficient * variable**next_degree / next_degree
    formula = sp.Add(antiderivative, integration_constant, evaluate=False)
    if str(sp.simplify(formula - result)) != "0":
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
    integration_constant = sp.Symbol("C")
    candidates = (
        (sp.sin(variable), -sp.cos(variable), "Use the sine antiderivative"),
        (sp.cos(variable), sp.sin(variable), "Use the cosine antiderivative"),
        (sp.exp(variable), sp.exp(variable), "Use the exponential antiderivative"),
    )
    for candidate_integrand, antiderivative, rule in candidates:
        if str(sp.simplify(integrand - candidate_integrand)) != "0":
            continue
        formula = sp.Add(antiderivative, integration_constant, evaluate=False)
        if str(sp.simplify(formula - result)) != "0":
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
    reverse_chain = _derive_reverse_chain_antiderivative(
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
    if len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    integration_constant = sp.Symbol("C")
    formula = sp.Add(*antiderivatives, integration_constant, evaluate=False)
    if str(sp.simplify(formula - result)) != "0":
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
    candidates = (
        (sp.sin(variable), -sp.cos(variable), "sine"),
        (sp.cos(variable), sp.sin(variable), "cosine"),
        (sp.exp(variable), sp.exp(variable), "exponential"),
    )
    for candidate, antiderivative, name in candidates:
        if str(sp.simplify(variable_part - candidate)) != "0":
            continue
        integration_constant = sp.Symbol("C")
        formula = sp.Add(
            sp.Mul(coefficient, antiderivative),
            integration_constant,
            evaluate=False,
        )
        if str(sp.simplify(formula - result)) != "0":
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
                rule=f"Use the {name} antiderivative",
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


def derive_function_substitution_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a substitution when an inner derivative multiplies a basic function."""
    function_candidates = (
        (sp.sin, _negative_cosine),
        (sp.cos, sp.sin),
        (sp.exp, sp.exp),
    )
    for factor in integrand.as_ordered_factors():
        for function, outer_antiderivative in function_candidates:
            if factor.func != function or len(factor.args) != 1:
                continue
            argument = factor.args[0]
            argument_derivative = sp.diff(argument, variable)
            if argument_derivative == sp.Integer(0):
                continue
            coefficient = sp.simplify(integrand / (factor * argument_derivative))
            if coefficient.has(variable):
                continue
            substitution_variable = sp.Symbol("u", real=True)
            transformed_function = function(substitution_variable)
            transformed_integral = BackendIntegral(
                integrand=transformed_function,
                variable=substitution_variable,
                coefficient=None if coefficient == sp.Integer(1) else coefficient,
            )
            formula_in_substitution_variable_term = outer_antiderivative(
                substitution_variable
            )
            formula_term = outer_antiderivative(argument)
            if coefficient != sp.Integer(1):
                formula_in_substitution_variable_term = sp.Mul(
                    coefficient,
                    formula_in_substitution_variable_term,
                    evaluate=False,
                )
                formula_term = sp.Mul(coefficient, formula_term, evaluate=False)
            integration_constant = sp.Symbol("C")
            formula_in_substitution_variable = sp.Add(
                formula_in_substitution_variable_term,
                integration_constant,
                evaluate=False,
            )
            formula = sp.Add(formula_term, integration_constant, evaluate=False)
            if str(sp.simplify(formula - result)) != "0":
                return ()
            return (
                BackendDerivationStep(
                    rule="Substitute the inner function",
                    before=BackendIntegral(integrand=integrand, variable=variable),
                    after=transformed_integral,
                    explanation=(
                        "The derivative of the inner function appears as a factor, so use the "
                        "inner function as the new variable."
                    ),
                    verification_method=VerificationMethod.SUBSTITUTION,
                    verification_detail=(
                        "Replacing the inner function and differential recovers the original "
                        "integrand."
                    ),
                    notes=(
                        BackendMathNote(
                            label="Choose the substitution",
                            expression=sp.Eq(
                                substitution_variable,
                                argument,
                                evaluate=False,
                            ),
                        ),
                        BackendMathNote(
                            label="Differentiate the substitution",
                            expression=BackendIdentity(
                                left=BackendDifferential(variable=substitution_variable),
                                right=BackendDifferential(
                                    variable=variable,
                                    coefficient=argument_derivative,
                                ),
                            ),
                        ),
                    ),
                ),
                BackendDerivationStep(
                    rule="Integrate in the new variable",
                    before=transformed_integral,
                    after=formula_in_substitution_variable,
                    explanation="Apply the basic antiderivative in the substitution variable.",
                    verification_method=VerificationMethod.DIFFERENTIATION,
                    verification_detail=(
                        "Differentiating with respect to the new variable recovers the "
                        "transformed integrand."
                    ),
                ),
                BackendDerivationStep(
                    rule="Substitute back",
                    before=formula_in_substitution_variable,
                    after=formula,
                    explanation="Replace the temporary variable with the original inner function.",
                    verification_method=VerificationMethod.SUBSTITUTION,
                    verification_detail="Direct substitution gives the exact antiderivative.",
                ),
            )
    return ()


def derive_integration_by_parts(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Use integration by parts when differentiating a polynomial factor simplifies it."""
    polynomial_factors = tuple(
        factor
        for factor in integrand.as_ordered_factors()
        if factor.has(variable) and factor.is_polynomial(variable)
    )
    if len(polynomial_factors) != 1:
        return ()
    chosen_u = polynomial_factors[0]
    polynomial = sp.Poly(chosen_u, variable)
    if polynomial.degree() < 1:
        return ()
    chosen_dv = sp.simplify(integrand / chosen_u)
    if (
        chosen_dv.func not in {sp.exp, sp.sin, sp.cos}
        or len(chosen_dv.args) != 1
        or sp.diff(chosen_dv.args[0], variable).has(variable)
    ):
        return ()
    chosen_v = sp.integrate(chosen_dv, variable)
    if chosen_v.has(sp.Integral):
        return ()
    chosen_du = sp.diff(chosen_u, variable)
    remaining_integrand = sp.simplify(chosen_v * chosen_du)
    remaining_antiderivative = sp.integrate(remaining_integrand, variable)
    integration_constant = sp.Symbol("C")
    expected = sp.Add(
        chosen_u * chosen_v,
        -remaining_antiderivative,
        integration_constant,
        evaluate=False,
    )
    if (
        remaining_antiderivative.has(sp.Integral)
        or sp.simplify(expected - result) != sp.Integer(0)
    ):
        return ()
    product_term = sp.simplify(chosen_u * chosen_v)
    if remaining_integrand.could_extract_minus_sign():
        by_parts_expression: BackendExpression = BackendSum(
            terms=(
                product_term,
                BackendIntegral(
                    integrand=-remaining_integrand,
                    variable=variable,
                ),
            )
        )
    else:
        by_parts_expression = BackendDifference(
            left=product_term,
            right=BackendIntegral(
                integrand=remaining_integrand,
                variable=variable,
            ),
        )
    return (
        BackendDerivationStep(
            rule="Choose integration by parts",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=by_parts_expression,
            explanation=(
                "Differentiate the polynomial factor because it becomes simpler, and "
                "antidifferentiate the other factor."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail=(
                "Expanding the integration-by-parts formula recovers the original integral."
            ),
            notes=(
                BackendMathNote(
                    label="Integration by parts",
                    expression=BackendIntegrationByPartsRule(),
                ),
                BackendMathNote(
                    label="Choose the algebraic part",
                    expression=BackendIdentity(
                        left=sp.Symbol("u"),
                        right=chosen_u,
                    ),
                ),
                BackendMathNote(
                    label="Choose the remaining differential",
                    expression=BackendIdentity(
                        left=sp.Symbol("dv"),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=(
                                None if chosen_dv == sp.Integer(1) else chosen_dv
                            ),
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Differentiate u",
                    expression=BackendIdentity(
                        left=sp.Symbol("du"),
                        right=BackendDifferential(
                            variable=variable,
                            coefficient=(
                                None if chosen_du == sp.Integer(1) else chosen_du
                            ),
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Antidifferentiate dv",
                    expression=BackendIdentity(
                        left=sp.Symbol("v"),
                        right=chosen_v,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the remaining integral",
            before=by_parts_expression,
            after=result,
            explanation=(
                "Integrate the simpler remaining term, combine the terms, and add the "
                "integration constant."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final expression recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Remaining antiderivative",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=remaining_integrand,
                            variable=variable,
                        ),
                        right=remaining_antiderivative,
                    ),
                ),
            ),
        ),
    )


def derive_trigonometric_power_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Reduce an even square of sine or cosine before integrating."""
    candidates = (
        (
            sp.sin(variable) ** 2,
            (sp.Integer(1) - sp.cos(2 * variable)) / 2,
            "sine",
        ),
        (
            sp.cos(variable) ** 2,
            (sp.Integer(1) + sp.cos(2 * variable)) / 2,
            "cosine",
        ),
    )
    for candidate, reduced, name in candidates:
        if sp.simplify(integrand - candidate) != sp.Integer(0):
            continue
        integration_constant = sp.Symbol("C")
        antiderivative = sp.Add(sp.integrate(reduced, variable), integration_constant)
        if sp.simplify(antiderivative - result) != sp.Integer(0):
            return ()
        return (
            BackendDerivationStep(
                rule=f"Use the {name} power-reduction identity",
                before=BackendIntegral(integrand=integrand, variable=variable),
                after=BackendIntegral(integrand=reduced, variable=variable),
                explanation=(
                    "Rewrite the squared trigonometric function using a double-angle "
                    "identity."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The power-reduction identity preserves the integrand.",
                notes=(
                    BackendMathNote(
                        label="Power-reduction identity",
                        expression=BackendIdentity(left=candidate, right=reduced),
                    ),
                ),
            ),
            BackendDerivationStep(
                rule="Integrate the reduced expression",
                before=BackendIntegral(integrand=reduced, variable=variable),
                after=result,
                explanation="Integrate the constant and double-angle cosine term separately.",
                verification_method=VerificationMethod.DIFFERENTIATION,
                verification_detail="Differentiating the result recovers the original square.",
            ),
        )
    return ()


def derive_gaussian_antiderivative(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Explain the standard special-function antiderivative of the Gaussian."""
    gaussian = sp.exp(-(variable**2))
    if sp.simplify(integrand - gaussian) != sp.Integer(0):
        return ()
    integration_constant = sp.Symbol("C")
    formula = sp.sqrt(sp.pi) * sp.erf(variable) / 2 + integration_constant
    if sp.simplify(formula - result) != sp.Integer(0):
        return ()
    generic_variable = sp.Symbol("t", real=True)
    definition_variable = sp.Symbol("z", real=True)
    return (
        BackendDerivationStep(
            rule="Express the antiderivative with the error function",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=formula,
            explanation=(
                "The Gaussian has no elementary antiderivative, so use the standard error "
                "function and include the integration constant."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the error-function expression gives the Gaussian exactly."
            ),
            notes=(
                BackendMathNote(
                    label="Definition of the error function",
                    expression=BackendIdentity(
                        left=sp.erf(definition_variable),
                        right=BackendProduct(
                            factors=(
                                2 / sp.sqrt(sp.pi),
                                BackendIntegral(
                                    integrand=sp.exp(-(generic_variable**2)),
                                    variable=generic_variable,
                                    lower=sp.Integer(0),
                                    upper=definition_variable,
                                ),
                            )
                        ),
                    ),
                ),
            ),
        ),
    )


def derive_square_root_rational_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Expose the hidden arctangent substitution in 1/(sqrt(x)(x+1))."""
    expected_integrand = 1 / (sp.sqrt(variable) * (variable + 1))
    if sp.simplify(integrand - expected_integrand) != sp.Integer(0):
        return ()
    substitution_variable = sp.Symbol("u", positive=True)
    transformed_integrand = 2 / (substitution_variable**2 + 1)
    transformed_integral = BackendIntegral(
        integrand=transformed_integrand,
        variable=substitution_variable,
    )
    integration_constant = sp.Symbol("C")
    transformed_result = 2 * sp.atan(substitution_variable) + integration_constant
    final_result = 2 * sp.atan(sp.sqrt(variable)) + integration_constant
    if sp.simplify(final_result - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Substitute the square root",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "The square root and its reciprocal suggest setting the new variable equal "
                "to the square root of x."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Replacing x and dx produces the displayed rational integral.",
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=sp.sqrt(variable),
                    ),
                ),
                BackendMathNote(
                    label="Rewrite x",
                    expression=BackendIdentity(
                        left=variable,
                        right=substitution_variable**2,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=substitution_variable,
                            coefficient=2 * substitution_variable,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the arctangent rule",
            before=transformed_integral,
            after=transformed_result,
            explanation="Apply the standard antiderivative of 1 divided by 1 plus u squared.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating gives the transformed integrand.",
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=1 / (substitution_variable**2 + 1),
                            variable=substitution_variable,
                        ),
                        right=sp.atan(substitution_variable) + integration_constant,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_result,
            after=final_result,
            explanation="Replace the temporary variable with the square root of x.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Direct substitution gives the final antiderivative.",
        ),
    )


def derive_shifted_semicircle_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Complete the square and use the standard semicircle antiderivative."""
    original_radicand = sp.simplify(integrand**2)
    expected_radicand = 2 * variable - variable**2
    if sp.simplify(original_radicand - expected_radicand) != sp.Integer(0):
        return ()
    shifted = variable - 1
    completed_radicand = 1 - shifted**2
    completed_integrand = sp.sqrt(completed_radicand)
    substitution_variable = sp.Symbol("u", real=True)
    transformed_integrand = sp.sqrt(1 - substitution_variable**2)
    integration_constant = sp.Symbol("C")
    transformed_formula = (
        substitution_variable * transformed_integrand / 2
        + sp.asin(substitution_variable) / 2
        + integration_constant
    )
    final_formula = (
        shifted * completed_integrand / 2
        + sp.asin(shifted) / 2
        + integration_constant
    )
    if sp.simplify(final_formula - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Complete the square under the radical",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=completed_integrand, variable=variable),
            explanation=(
                "Rewrite the quadratic as one minus a shifted square so it matches the "
                "standard semicircle form."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Expanding the completed square recovers the original radicand.",
            notes=(
                BackendMathNote(
                    label="Completed square",
                    expression=BackendIdentity(
                        left=expected_radicand,
                        right=completed_radicand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Shift the variable",
            before=BackendIntegral(integrand=completed_integrand, variable=variable),
            after=BackendIntegral(
                integrand=transformed_integrand,
                variable=substitution_variable,
            ),
            explanation="Set u equal to the shifted x-expression; its differential is dx.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The translation changes only the variable name.",
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=shifted,
                    ),
                ),
                BackendMathNote(
                    label="Differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=substitution_variable),
                        right=BackendDifferential(variable=variable),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the semicircle antiderivative",
            before=BackendIntegral(
                integrand=transformed_integrand,
                variable=substitution_variable,
            ),
            after=transformed_formula,
            explanation=(
                "Apply the standard antiderivative for the upper unit semicircle."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the formula gives the semicircle integrand.",
            notes=(
                BackendMathNote(
                    label="Standard rule",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=transformed_integrand,
                            variable=substitution_variable,
                        ),
                        right=transformed_formula,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_formula,
            after=final_formula,
            explanation="Replace u with x minus one.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Substitution returns the result to the original variable.",
        ),
    )


def derive_inverse_hyperbolic_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Normalize 1/sqrt(a*x^2+b) to the inverse-hyperbolic-sine rule."""
    radicand = sp.simplify(integrand**-2)
    try:
        polynomial = sp.Poly(radicand, variable)
    except sp.PolynomialError:
        return ()
    if polynomial.degree() != _QUADRATIC_DEGREE:
        return ()
    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    if (
        coefficient_b != sp.Integer(0)
        or coefficient_a.is_positive is not True
        or coefficient_c.is_positive is not True
        or sp.simplify(integrand - 1 / sp.sqrt(radicand)) != sp.Integer(0)
    ):
        return ()
    substitution_variable = sp.Symbol("u", real=True)
    scale = sp.sqrt(coefficient_a / coefficient_c)
    coefficient = 1 / sp.sqrt(coefficient_a)
    transformed_integrand = 1 / sp.sqrt(substitution_variable**2 + 1)
    transformed_integral = BackendIntegral(
        integrand=transformed_integrand,
        variable=substitution_variable,
        coefficient=coefficient,
    )
    integration_constant = sp.Symbol("C")
    transformed_formula = coefficient * sp.asinh(substitution_variable) + integration_constant
    final_formula = coefficient * sp.asinh(scale * variable) + integration_constant
    if sp.simplify(final_formula - result) != sp.Integer(0):
        return ()
    return (
        BackendDerivationStep(
            rule="Normalize the quadratic radical",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=transformed_integral,
            explanation=(
                "Scale the variable so the expression under the square root becomes one plus "
                "u squared."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The variable scaling produces the normalized radical exactly.",
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=BackendIdentity(
                        left=substitution_variable,
                        right=scale * variable,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=substitution_variable,
                            coefficient=1 / scale,
                        ),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Use the inverse hyperbolic sine rule",
            before=transformed_integral,
            after=transformed_formula,
            explanation=(
                "The normalized integrand is the derivative of inverse hyperbolic sine."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating inverse hyperbolic sine gives the normalized integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Rule to remember",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=transformed_integrand,
                            variable=substitution_variable,
                        ),
                        right=sp.asinh(substitution_variable) + integration_constant,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Substitute back",
            before=transformed_formula,
            after=final_formula,
            explanation="Replace the temporary variable with its scaled x-expression.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Substitution gives the exact antiderivative in x.",
        ),
    )


def derive_partial_fraction_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Integrate rational functions whose partial fractions avoid logarithmic domains."""
    if not integrand.is_rational_function(variable) or result.has(sp.log):
        return ()
    decomposition = sp.apart(integrand, variable)
    terms = tuple(decomposition.as_ordered_terms())
    if str(decomposition) == str(integrand) or len(terms) < _MINIMUM_PRODUCT_FACTORS:
        return ()
    antiderivatives = tuple(sp.integrate(term, variable) for term in terms)
    if any(item.has(sp.Integral) for item in antiderivatives):
        return ()
    return (
        BackendDerivationStep(
            rule="Decompose into partial fractions",
            before=BackendIntegral(integrand=integrand, variable=variable),
            after=BackendIntegral(integrand=decomposition, variable=variable),
            explanation=(
                "Rewrite the rational function as a sum of simpler fractions that can be "
                "integrated separately."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="Combining the partial fractions recovers the original fraction.",
            notes=(
                BackendMathNote(
                    label="Partial-fraction identity",
                    expression=BackendIdentity(left=integrand, right=decomposition),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Integrate each partial fraction",
            before=BackendIntegral(integrand=decomposition, variable=variable),
            after=result,
            explanation="Integrate the separated power and arctangent terms, then combine them.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the final result recovers the rational function.",
            notes=tuple(
                BackendMathNote(
                    label=f"Term {index}",
                    expression=BackendIdentity(
                        left=BackendIntegral(integrand=term, variable=variable),
                        right=antiderivative,
                    ),
                )
                for index, (term, antiderivative) in enumerate(
                    zip(terms, antiderivatives, strict=True),
                    start=1,
                )
            ),
        ),
    )


def derive_reciprocal_quadratic_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive a completed-square arctangent integral when applicable."""
    numerator, denominator = sp.fraction(sp.together(integrand))
    if numerator.has(variable):
        return ()
    polynomial = sp.Poly(denominator, variable)
    if polynomial.degree() != _QUADRATIC_DEGREE:
        return ()
    coefficient_a = polynomial.coeff_monomial(variable**2)
    coefficient_b = polynomial.coeff_monomial(variable)
    coefficient_c = polynomial.coeff_monomial(1)
    center = sp.simplify(-coefficient_b / (2 * coefficient_a))
    radius_squared = sp.simplify(
        coefficient_c / coefficient_a - coefficient_b**2 / (4 * coefficient_a**2)
    )
    if radius_squared.is_positive is not True:
        return ()
    prefactor = sp.simplify(numerator / coefficient_a)
    shift = (
        variable
        if center == sp.Integer(0)
        else sp.Add(variable, -center, evaluate=False)
    )
    completed_denominator = sp.Add(
        sp.Pow(shift, 2, evaluate=False),
        radius_squared,
        evaluate=False,
    )
    completed_integrand = sp.Mul(
        prefactor,
        sp.Pow(completed_denominator, -1, evaluate=False),
        evaluate=False,
    )
    if str(sp.simplify(integrand - completed_integrand)) != "0":
        message = "completing the square changed the integrand"
        raise ValueError(message)
    radius = sp.sqrt(radius_squared)
    radius_scale = sp.sqrt(sp.simplify(4 * radius_squared))
    raw_normalized_argument = sp.Mul(
        sp.expand(2 * shift),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    raw_normalized_coefficient = sp.Mul(
        sp.expand(2 * prefactor),
        sp.Pow(radius_scale, -1, evaluate=False),
        evaluate=False,
    )
    if isinstance(radius_scale, sp.Rational):
        normalized_argument = sp.cancel(raw_normalized_argument)
        normalized_coefficient = sp.cancel(raw_normalized_coefficient)
    else:
        normalized_argument = raw_normalized_argument
        normalized_coefficient = raw_normalized_coefficient
    coefficient_is_one = sp.simplify(normalized_coefficient - 1) == sp.Integer(0)
    substitution_variable = sp.Symbol("u", real=True)
    unit_denominator = sp.Add(
        sp.Pow(substitution_variable, 2, evaluate=False),
        sp.Integer(1),
        evaluate=False,
    )
    unit_integrand = sp.Pow(unit_denominator, -1, evaluate=False)
    integration_constant = sp.Symbol("C")
    formula_in_substitution_variable_term = (
        sp.atan(substitution_variable)
        if coefficient_is_one
        else sp.Mul(
            normalized_coefficient,
            sp.atan(substitution_variable),
            evaluate=False,
        )
    )
    formula_in_substitution_variable = sp.Add(
        formula_in_substitution_variable_term,
        integration_constant,
        evaluate=False,
    )
    formula_term = (
        sp.atan(normalized_argument)
        if coefficient_is_one
        else sp.Mul(
            normalized_coefficient,
            sp.atan(normalized_argument),
            evaluate=False,
        )
    )
    formula = sp.Add(
        formula_term,
        integration_constant,
        evaluate=False,
    )
    transformed_integrand = sp.Mul(
        normalized_coefficient,
        unit_integrand.subs(substitution_variable, normalized_argument),
        sp.diff(normalized_argument, variable),
    )
    if str(sp.simplify(transformed_integrand - completed_integrand)) != "0":
        message = "the substitution changed the completed-square integral"
        raise ValueError(message)
    if str(
        sp.simplify(
            sp.diff(formula_in_substitution_variable, substitution_variable)
            - normalized_coefficient * unit_integrand
        )
    ) != "0":
        message = "the arctangent formula failed differentiation verification"
        raise ValueError(message)
    if str(sp.simplify(formula - result)) != "0":
        message = "the simplified antiderivative differs from the exact result"
        raise ValueError(message)
    return _build_reciprocal_quadratic_steps(
        integrand=integrand,
        variable=variable,
        denominator=denominator,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        coefficient_c=coefficient_c,
        radius_squared=radius_squared,
        completed_denominator=completed_denominator,
        completed_integrand=completed_integrand,
        shift=shift,
        radius=radius,
        substitution_variable=substitution_variable,
        unit_integrand=unit_integrand,
        normalized_argument=normalized_argument,
        normalized_coefficient=normalized_coefficient,
        coefficient_is_one=coefficient_is_one,
        formula_in_substitution_variable=formula_in_substitution_variable,
        formula=formula,
        integration_constant=integration_constant,
    )


def derive_definite_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Apply the Fundamental Theorem to a proper elementary definite integral."""
    if lower in {sp.oo, -sp.oo} or upper in {sp.oo, -sp.oo}:
        return ()
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    upper_value = sp.simplify(antiderivative.subs(variable, upper))
    lower_value = sp.simplify(antiderivative.subs(variable, lower))
    if sp.simplify(upper_value - lower_value - result) != sp.Integer(0):
        return ()
    evaluated_at_bounds = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    endpoint_difference = BackendDifference(left=upper_value, right=lower_value)
    generic_variable = sp.Symbol("x", real=True)
    generic_lower = sp.Symbol("a", real=True)
    generic_upper = sp.Symbol("b", real=True)
    generic_function = sp.Function("f")(generic_variable)
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=evaluated_at_bounds,
            explanation=(
                "Find an antiderivative, then evaluate it at the upper bound minus the "
                "lower bound."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the chosen antiderivative recovers the integrand."
            ),
            notes=(
                BackendMathNote(
                    label="Fundamental Theorem",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=generic_function,
                            variable=generic_variable,
                            lower=generic_lower,
                            upper=generic_upper,
                        ),
                        right=BackendDifference(
                            left=sp.Function("F")(generic_upper),
                            right=sp.Function("F")(generic_lower),
                        ),
                    ),
                ),
                BackendMathNote(
                    label="Chosen antiderivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=antiderivative,
                            variable=variable,
                        ),
                        right=integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the bounds",
            before=evaluated_at_bounds,
            after=endpoint_difference,
            explanation=(
                "Substitute the upper and lower bounds into the antiderivative, keeping "
                "upper minus lower."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="Both endpoint values were evaluated exactly.",
            notes=(
                BackendMathNote(
                    label="Upper bound",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtBounds(
                            expression=antiderivative,
                            variable=variable,
                            lower=upper,
                            upper=upper,
                        ),
                        right=upper_value,
                    ),
                ),
                BackendMathNote(
                    label="Lower bound",
                    expression=BackendIdentity(
                        left=BackendEvaluationAtBounds(
                            expression=antiderivative,
                            variable=variable,
                            lower=lower,
                            upper=lower,
                        ),
                        right=lower_value,
                    ),
                ),
            ),
        ),
    ]
    if str(endpoint_difference) != str(result):
        steps.append(
            BackendDerivationStep(
                rule="Finish the arithmetic",
                before=endpoint_difference,
                after=result,
                explanation="Subtract the lower-bound value from the upper-bound value.",
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="The final subtraction was evaluated exactly.",
            )
        )
    return tuple(steps)


def derive_improper_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Evaluate a convergent one-ended improper integral through its defining limit."""
    has_infinite_lower = lower == -sp.oo and upper not in {sp.oo, -sp.oo}
    has_infinite_upper = upper == sp.oo and lower not in {sp.oo, -sp.oo}
    if not has_infinite_lower and not has_infinite_upper:
        return ()
    bound = sp.Symbol("a" if has_infinite_lower else "b", real=True)
    finite_lower = bound if has_infinite_lower else lower
    finite_upper = upper if has_infinite_lower else bound
    approach_point = -sp.oo if has_infinite_lower else sp.oo
    antiderivative = sp.integrate(integrand, variable)
    if antiderivative.has(sp.Integral):
        return ()
    if sp.simplify(sp.diff(antiderivative, variable) - integrand) != sp.Integer(0):
        return ()
    limit_of_integral = BackendLimit(
        expression=BackendIntegral(
            integrand=integrand,
            variable=variable,
            lower=finite_lower,
            upper=finite_upper,
        ),
        variable=bound,
        point=approach_point,
    )
    evaluated_at_bounds = BackendEvaluationAtBounds(
        expression=antiderivative,
        variable=variable,
        lower=finite_lower,
        upper=finite_upper,
    )
    limit_of_antiderivative = BackendLimit(
        expression=evaluated_at_bounds,
        variable=bound,
        point=approach_point,
    )
    return (
        BackendDerivationStep(
            rule="Rewrite the improper integral as a limit",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=limit_of_integral,
            explanation=(
                "Replace the infinite endpoint with a finite bound, then let that bound "
                "approach infinity."
            ),
            verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
            verification_detail="This limit is the definition of the improper integral.",
        ),
        BackendDerivationStep(
            rule="Apply the Fundamental Theorem of Calculus",
            before=limit_of_integral,
            after=limit_of_antiderivative,
            explanation=(
                "Evaluate the finite integral with an antiderivative before taking the limit."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail="Differentiating the antiderivative recovers the integrand.",
            notes=(
                BackendMathNote(
                    label="Chosen antiderivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=antiderivative,
                            variable=variable,
                        ),
                        right=integrand,
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Evaluate the limit",
            before=limit_of_antiderivative,
            after=result,
            explanation=(
                "Evaluate the finite endpoint and take the limit at the infinite endpoint."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The endpoint limit gives the exact convergent value.",
        ),
    )


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
            numerator_leading = numerator_polynomial.coeff_monomial(
                variable**numerator_degree
            ) * variable**numerator_degree
            denominator_leading = denominator_polynomial.coeff_monomial(
                variable**denominator_degree
            ) * variable**denominator_degree
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
    standard_sine_quotient = sp.sin(variable) / variable
    if (
        point == sp.Integer(0)
        and sp.simplify(expression - standard_sine_quotient) == sp.Integer(0)
    ):
        generic_variable = sp.Symbol("u", real=True)
        return (
            BackendDerivationStep(
                rule="Use the standard sine limit",
                before=displayed_limit,
                after=result,
                explanation=(
                    "This expression is the standard trigonometric limit at zero."
                ),
                verification_method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                verification_detail="The expression exactly matches the standard sine limit.",
                notes=(
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

    numerator, denominator = sp.fraction(sp.together(expression))
    if point == sp.Integer(0) and denominator == variable and numerator.func == sp.sin:
        sine_argument = numerator.args[0]
        frequency = sp.simplify(sine_argument / variable)
        if not frequency.has(variable) and frequency != sp.Integer(0) and result == frequency:
            generic_variable = sp.Symbol("u", real=True)
            return (
                BackendDerivationStep(
                    rule="Normalize to the standard sine limit",
                    before=displayed_limit,
                    after=result,
                    explanation=(
                        "Multiply and divide by the sine argument's coefficient, then use the "
                        "standard sine limit."
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
                                        frequency,
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


def derive_limit(
    expression: sp.Basic,
    variable: sp.Symbol,
    point: sp.Basic,
    direction: str | None,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive familiar limits with the shortest standard student method."""
    sine_derivation = _derive_sine_limit(expression, variable, point, direction, result)
    if sine_derivation:
        return sine_derivation
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
        if (
            numerator_at_point == sp.Integer(0)
            and denominator_at_point == sp.Integer(0)
        ):
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
        if (
            substituted == result
            and not substituted.has(sp.zoo)
            and not substituted.has(sp.nan)
        ):
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


def derive_dirichlet_integral(
    integrand: sp.Basic,
    variable: sp.Symbol,
    lower: sp.Basic,
    upper: sp.Basic,
    result: sp.Basic,
) -> tuple[BackendDerivationStep, ...]:
    """Derive the classical Dirichlet integral with a damping parameter."""
    scaled_sine = sp.simplify(integrand * variable)
    if scaled_sine.func != sp.sin or len(scaled_sine.args) != 1:
        return ()
    sine_argument = scaled_sine.args[0]
    frequency = sp.simplify(sine_argument / variable)
    if frequency.has(variable) or frequency.is_positive is not True:
        return ()
    if str(sp.simplify(lower)) != "0" or upper != sp.oo:
        return ()
    if str(sp.simplify(result - sp.pi / 2)) != "0":
        return ()
    expected_integrand = sp.sin(variable) / variable
    if frequency != sp.Integer(1):
        transformed_variable = sp.Symbol("t", positive=True)
        transformed_integral = BackendIntegral(
            integrand=sp.sin(transformed_variable) / transformed_variable,
            variable=transformed_variable,
            lower=lower,
            upper=upper,
        )
        substitution_step = BackendDerivationStep(
            rule="Scale the integration variable",
            before=BackendIntegral(
                integrand=integrand,
                variable=variable,
                lower=lower,
                upper=upper,
            ),
            after=transformed_integral,
            explanation=(
                "Scale the variable so the sine argument becomes the new integration variable; "
                "the scale factors cancel."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the variable and differential produces the standard Dirichlet "
                "integral."
            ),
            notes=(
                BackendMathNote(
                    label="Choose the substitution",
                    expression=sp.Eq(
                        transformed_variable,
                        sine_argument,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Rewrite the original variable",
                    expression=sp.Eq(
                        variable,
                        transformed_variable / frequency,
                        evaluate=False,
                    ),
                ),
                BackendMathNote(
                    label="Change the differential",
                    expression=BackendIdentity(
                        left=BackendDifferential(variable=variable),
                        right=BackendDifferential(
                            variable=transformed_variable,
                            coefficient=1 / frequency,
                        ),
                    ),
                ),
            ),
        )
        standard_steps = derive_dirichlet_integral(
            sp.sin(transformed_variable) / transformed_variable,
            transformed_variable,
            lower,
            upper,
            result,
        )
        return (substitution_step, *standard_steps)

    parameter = sp.Symbol("a", positive=True)
    function = sp.Function("F")(parameter)
    integration_constant = sp.Symbol("C")
    damping = sp.exp(-parameter * variable)
    damped_integrand = damping * expected_integrand
    laplace_integrand = damping * sp.sin(variable)
    derivative_value = -1 / (parameter**2 + 1)
    general_function = sp.Add(
        integration_constant,
        -sp.atan(parameter),
        evaluate=False,
    )
    resolved_function = sp.Add(
        sp.pi / 2,
        -sp.atan(parameter),
        evaluate=False,
    )
    original_integral = BackendIntegral(
        integrand=integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    damped_integral = BackendIntegral(
        integrand=damped_integrand,
        variable=variable,
        lower=lower,
        upper=upper,
    )
    parameter_limit = BackendLimit(
        expression=function,
        variable=parameter,
        point=sp.Integer(0),
        direction="+",
    )
    derivative_identity = BackendIdentity(
        left=BackendDerivative(expression=function, variable=parameter),
        right=derivative_value,
    )
    return (
        BackendDerivationStep(
            rule="Introduce a damping parameter",
            before=original_integral,
            after=parameter_limit,
            explanation=(
                "Temporarily multiply by an exponential damping factor. This makes the "
                "parameterized integral easier to differentiate, and the original integral "
                "is recovered as the damping disappears."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "The Abel limit of the damped integral equals the original improper integral."
            ),
            notes=(
                BackendMathNote(
                    label="Define the damped integral",
                    expression=BackendIdentity(left=function, right=damped_integral),
                ),
                BackendMathNote(
                    label="Remove the damping at the end",
                    expression=BackendIdentity(left=original_integral, right=parameter_limit),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Differentiate with respect to the parameter",
            before=BackendIdentity(left=function, right=damped_integral),
            after=derivative_identity,
            explanation=(
                "Differentiating the damping factor cancels the division by the integration "
                "variable, leaving a standard Laplace integral."
            ),
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating under the integral sign produces the displayed Laplace integral."
            ),
            notes=(
                BackendMathNote(
                    label="Differentiate the integrand",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=damped_integrand,
                            variable=parameter,
                        ),
                        right=-laplace_integrand,
                    ),
                ),
                BackendMathNote(
                    label="Evaluate the remaining Laplace integral",
                    expression=BackendIdentity(
                        left=BackendIntegral(
                            integrand=laplace_integrand,
                            variable=variable,
                            lower=lower,
                            upper=upper,
                        ),
                        right=1 / (parameter**2 + 1),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Recover the parameterized integral",
            before=derivative_identity,
            after=BackendIdentity(left=function, right=general_function),
            explanation="Integrate with respect to the parameter to recover the function.",
            verification_method=VerificationMethod.DIFFERENTIATION,
            verification_detail=(
                "Differentiating the recovered expression gives the parameter derivative."
            ),
            notes=(
                BackendMathNote(
                    label="Arctangent derivative",
                    expression=BackendIdentity(
                        left=BackendDerivative(
                            expression=sp.atan(parameter),
                            variable=parameter,
                        ),
                        right=1 / (parameter**2 + 1),
                    ),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Determine the constant",
            before=BackendIdentity(left=function, right=general_function),
            after=BackendIdentity(left=function, right=resolved_function),
            explanation=(
                "As the damping becomes infinitely strong, the integral tends to zero. Use "
                "that boundary value to determine the integration constant."
            ),
            verification_method=VerificationMethod.EXACT_ARITHMETIC,
            verification_detail="The boundary limits force the constant to equal pi over two.",
            notes=(
                BackendMathNote(
                    label="The damped integral vanishes",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=function,
                            variable=parameter,
                            point=sp.oo,
                        ),
                        right=sp.Integer(0),
                    ),
                ),
                BackendMathNote(
                    label="Arctangent limit",
                    expression=BackendIdentity(
                        left=BackendLimit(
                            expression=sp.atan(parameter),
                            variable=parameter,
                            point=sp.oo,
                        ),
                        right=sp.pi / 2,
                    ),
                ),
                BackendMathNote(
                    label="Therefore",
                    expression=sp.Eq(integration_constant, sp.pi / 2, evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Remove the damping",
            before=parameter_limit,
            after=result,
            explanation=(
                "Let the damping parameter approach zero from the positive side to recover "
                "the original improper integral."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="The right-hand limit of the resolved expression is pi over two.",
            notes=(
                BackendMathNote(
                    label="Use the resolved formula",
                    expression=BackendIdentity(
                        left=parameter_limit,
                        right=BackendLimit(
                            expression=resolved_function,
                            variable=parameter,
                            point=sp.Integer(0),
                            direction="+",
                        ),
                    ),
                ),
            ),
        ),
    )
