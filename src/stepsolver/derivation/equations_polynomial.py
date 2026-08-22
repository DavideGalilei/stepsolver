"""Human-readable quadratic and higher-degree equation methods."""

from __future__ import annotations

from math import ceil, floor

import sympy as sp

from stepsolver.derivation.equations_support import (
    append_if_changed,
    equivalent_step,
    newton_values,
    rational_root_candidates,
    relations,
)
from stepsolver.derivation.model import (
    BackendApproximateSolutions,
    BackendCardanoSolution,
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendNewtonIterations,
    BackendNewtonRule,
    BackendQuadraticSolutions,
    EquationBackendExpression,
)
from stepsolver.results import VerificationMethod

_DISPLAY_DIGITS = 7
_MAX_RATIONAL_ROOT_CANDIDATES = 12


def derive_quadratic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Solve a quadratic with factoring or the quadratic formula."""
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
        current = append_if_changed(
            steps,
            rule="Factor the quadratic",
            before=current,
            after=factored_equation,
            explanation="Rewrite the quadratic as a product of linear factors.",
            variable=variable,
            excluded_values=excluded_values,
        )
        factor_equations = tuple(sp.Eq(factor, 0) for factor in factor_bases)
        separated_factors: EquationBackendExpression = (
            factor_equations[0] if len(factor_equations) == 1 else factor_equations
        )
        current = append_if_changed(
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
            excluded_values=excluded_values,
        )
        factor_roots = tuple(
            -sp.Poly(factor, variable).coeff_monomial(1)
            / sp.Poly(factor, variable).coeff_monomial(variable)
            for factor in factor_bases
        )
        current = append_if_changed(
            steps,
            rule="Solve each factor",
            before=current,
            after=relations(variable, factor_roots),
            explanation="Solve each resulting linear equation and combine the solutions.",
            variable=variable,
            excluded_values=excluded_values,
        )
        if set(map(str, factor_roots)) != set(map(str, roots)):
            append_if_changed(
                steps,
                rule="Apply the domain restrictions",
                before=current,
                after=relations(variable, roots),
                explanation=(
                    "Discard every candidate that makes an original denominator equal to zero."
                ),
                variable=variable,
                excluded_values=excluded_values,
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


def _factored_polynomial_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    expression = polynomial.as_expr()
    factored = sp.factor(expression)
    if str(factored) == str(expression) or not (factored.is_Mul or factored.is_Pow):
        return ()
    _coefficient, factor_pairs = sp.factor_list(expression, variable)
    factor_bases = tuple(factor for factor, _multiplicity in factor_pairs if factor.has(variable))
    if not factor_bases:
        return ()
    steps: list[BackendDerivationStep] = []
    factored_equation = sp.Eq(factored, 0, evaluate=False)
    current: EquationBackendExpression = append_if_changed(
        steps,
        rule="Factor the polynomial",
        before=equation,
        after=factored_equation,
        explanation="Rewrite the polynomial as a product of lower-degree factors.",
        variable=variable,
        excluded_values=excluded_values,
    )
    factor_equations = tuple(sp.Eq(factor, 0, evaluate=False) for factor in factor_bases)
    separated: EquationBackendExpression = (
        factor_equations[0] if len(factor_equations) == 1 else factor_equations
    )
    current = append_if_changed(
        steps,
        rule="Apply the zero-product property",
        before=current,
        after=separated,
        explanation="At least one factor must equal zero, so solve each factor separately.",
        variable=variable,
        excluded_values=excluded_values,
    )
    append_if_changed(
        steps,
        rule="Keep the real solutions",
        before=current,
        after=relations(variable, roots),
        explanation=(
            "Solve the lower-degree equations and keep the real values allowed by the "
            "original domain."
        ),
        variable=variable,
        excluded_values=excluded_values,
    )
    return tuple(steps)


def derive_cubic_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Solve a cubic with rational roots, Cardano, or Newton iteration."""
    factored = _factored_polynomial_steps(
        equation,
        variable,
        polynomial,
        roots,
        excluded_values,
    )
    if factored:
        return factored
    expression = polynomial.as_expr()
    coefficient_a = polynomial.coeff_monomial(variable**3)
    coefficient_b = polynomial.coeff_monomial(variable**2)
    coefficient_c = polynomial.coeff_monomial(variable)
    coefficient_d = polynomial.coeff_monomial(1)
    depressed_linear = sp.simplify(
        (3 * coefficient_a * coefficient_c - coefficient_b**2) / (3 * coefficient_a**2)
    )
    depressed_constant = sp.simplify(
        (
            2 * coefficient_b**3
            - 9 * coefficient_a * coefficient_b * coefficient_c
            + 27 * coefficient_a**2 * coefficient_d
        )
        / (27 * coefficient_a**3)
    )
    discriminant = sp.simplify((depressed_constant / 2) ** 2 + (depressed_linear / 3) ** 3)
    if discriminant.is_positive is not True or len(roots) != 1:
        return ()
    shift = sp.simplify(-coefficient_b / (3 * coefficient_a))
    first_radicand = sp.simplify(-depressed_constant / 2 + sp.sqrt(discriminant))
    second_radicand = sp.simplify(-depressed_constant / 2 - sp.sqrt(discriminant))
    cardano_solution = BackendCardanoSolution(
        variable=variable,
        shift=shift,
        first_radicand=first_radicand,
        second_radicand=second_radicand,
    )
    candidates = rational_root_candidates(polynomial, variable)
    function = sp.Function("f")
    candidate_checks = tuple(
        sp.Eq(
            function(candidate),
            sp.simplify(expression.subs(variable, candidate)),
            evaluate=False,
        )
        for candidate in candidates
    )
    numerical_root = sp.N(roots[0], _DISPLAY_DIGITS)
    numerical_root_float = float(str(numerical_root))
    lower = sp.Integer(floor(numerical_root_float))
    upper = sp.Integer(ceil(numerical_root_float))
    bracket_checks = tuple(
        sp.Eq(
            function(point),
            sp.simplify(expression.subs(variable, point)),
            evaluate=False,
        )
        for point in (lower, upper)
    )
    derivative = sp.diff(expression, variable)
    initial = lower if derivative.subs(variable, lower) != sp.Integer(0) else upper
    notes: list[BackendMathNote] = []
    if candidate_checks and len(candidate_checks) <= _MAX_RATIONAL_ROOT_CANDIDATES:
        notes.append(
            BackendMathNote(
                label="Rational-root test",
                expression=candidate_checks,
            )
        )
    notes.extend(
        (
            BackendMathNote(label="Bracket the root", expression=bracket_checks),
            BackendMathNote(label="Newton iteration", expression=BackendNewtonRule()),
            BackendMathNote(
                label="Successive estimates",
                expression=BackendNewtonIterations(
                    variable=variable,
                    values=newton_values(expression, variable, initial),
                ),
            ),
            BackendMathNote(label="Exact form (optional)", expression=cardano_solution),
        )
    )
    return (
        BackendDerivationStep(
            rule="Approximate the real root",
            before=equation,
            after=BackendApproximateSolutions(
                variable=variable,
                roots=(numerical_root,),
            ),
            explanation=(
                "None of the rational-root candidates is a root, so grouping does not give "
                "a rational factorization. Bracket the one real root, then refine it with "
                "Newton's method."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Substitution gives a residual consistent with the displayed precision; the "
                "exact Cardano form is retained as an optional note."
            ),
            notes=tuple(notes),
        ),
    )


def derive_generic_polynomial_steps(
    equation: sp.Equality,
    variable: sp.Symbol,
    polynomial: sp.Poly,
    roots: tuple[sp.Basic, ...],
    excluded_values: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Solve higher-degree polynomials through verified factoring when possible."""
    factored = _factored_polynomial_steps(
        equation,
        variable,
        polynomial,
        roots,
        excluded_values,
    )
    if factored:
        return factored
    if not roots:
        return (
            equivalent_step(
                rule="Check for real roots",
                before=equation,
                after=(),
                explanation=(
                    "The polynomial does not cross zero on the real number line, so it has "
                    "no real solutions."
                ),
                variable=variable,
                excluded_values=excluded_values,
            ),
        )
    approximations = tuple(sp.N(root, 7) for root in roots)
    return (
        BackendDerivationStep(
            rule="Approximate the real roots",
            before=equation,
            after=BackendApproximateSolutions(variable=variable, roots=approximations),
            explanation=(
                "The polynomial has no simpler exact factorization. Bracket each real root, "
                "then refine it with Newton's method."
            ),
            verification_method=VerificationMethod.BACKEND_IDENTITY,
            verification_detail=(
                "Substitution gives a residual consistent with the displayed precision; "
                "the exact algebraic roots remain in the result."
            ),
            notes=(BackendMathNote(label="Newton iteration", expression=BackendNewtonRule()),),
        ),
    )
