"""Human-readable derivations for logarithmic and exponential equations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import gcd
from typing import cast

import sympy as sp

from stepsolver.derivation.equations import derive_polynomial_equation
from stepsolver.derivation.model import (
    BackendDerivationStep,
    BackendIdentity,
    BackendMathNote,
    BackendStepConstraint,
)
from stepsolver.results import VerificationMethod

_QUADRATIC_DEGREE = 2
_SSTR = cast("Callable[[object], str]", getattr(sp, "sstr"))


def _log_argument(log_term: sp.Basic) -> sp.Basic:
    return log_term.args[0]


def _is_basic_sequence(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return False
    return all(isinstance(item, sp.Basic) for item in cast("Sequence[object]", value))


def _solve_candidates(equation: sp.Equality, variable: sp.Symbol) -> tuple[sp.Basic, ...] | None:
    """Return a nonempty sequence of exact candidate values."""
    try:
        solved = sp.solve(equation, variable)
    except (NotImplementedError, TypeError, ValueError):
        return None
    if not _is_basic_sequence(solved) or not solved:
        return None
    return tuple(dict.fromkeys(cast("Sequence[sp.Basic]", solved)))


def _log_arguments_are_positive(
    equation: sp.Equality,
    variable: sp.Symbol,
    candidate: sp.Basic,
) -> bool:
    """Check that every logarithm keeps a positive argument at a candidate."""
    for log_term in equation.atoms(sp.log):
        argument = _log_argument(log_term).subs(variable, candidate)
        if sp.simplify(argument).is_positive is not True:
            return False
    return True


def _satisfies_original(
    equation: sp.Equality,
    variable: sp.Symbol,
    candidate: sp.Basic,
) -> bool:
    """Check a candidate in the original equation and its logarithm domain."""
    if not _log_arguments_are_positive(equation, variable, candidate):
        return False
    residual = equation.lhs.subs(variable, candidate) - equation.rhs.subs(variable, candidate)
    try:
        return sp.simplify(residual) == sp.Integer(0)
    except (TypeError, ValueError):
        return False


def _roots_match(valid: tuple[sp.Basic, ...], roots: tuple[sp.Basic, ...]) -> bool:
    return {str(sp.simplify(root)) for root in roots} == {
        str(sp.simplify(candidate)) for candidate in valid
    }


def _log_domain_constraints(log_terms: tuple[sp.Basic, ...]) -> tuple[BackendStepConstraint, ...]:
    seen: dict[str, BackendStepConstraint] = {}
    for log_term in log_terms:
        argument = _log_argument(log_term)
        key = str(argument)
        seen[key] = BackendStepConstraint(
            explanation="A logarithm requires a positive argument.",
            expression=sp.Gt(argument, sp.Integer(0), evaluate=False),
        )
    return tuple(seen.values())


def _log_parts(term: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """Return ``(base, argument)`` for one logarithm written in any base."""
    if term.func == sp.log and len(term.args) == 1:
        return sp.E, term.args[0]
    if not term.is_Mul:
        return None
    logarithms = [factor for factor in term.args if factor.func == sp.log and len(factor.args) == 1]
    if len(logarithms) != 1:
        return None
    logarithm = logarithms[0]
    coefficient = sp.Mul(*(factor for factor in term.args if factor is not logarithm))
    if coefficient == sp.Integer(0):
        return None
    base = sp.simplify(sp.exp(1 / coefficient))
    if base.is_positive is not True or base == sp.Integer(1):
        return None
    if sp.simplify(coefficient * sp.log(base) - 1) != sp.Integer(0):
        return None
    return base, logarithm.args[0]


def _split_log_sum(expression: sp.Basic) -> tuple[sp.Basic, sp.Basic] | None:
    """Combine ``log(A) + log(B)`` or ``log(A) - log(B)`` into one argument."""
    terms = expression.as_ordered_terms()
    if len(terms) != 2:
        return None
    parsed: list[tuple[tuple[sp.Basic, sp.Basic], int]] = []
    for term in terms:
        sign = 1
        candidate = term
        if candidate.could_extract_minus_sign():
            positive = -candidate
            if _log_parts(positive) is not None:
                candidate = positive
                sign = -1
        parts = _log_parts(candidate)
        if parts is None:
            return None
        parsed.append((parts, sign))
    first, second = parsed
    (first_base, first_argument), first_sign = first
    (second_base, second_argument), second_sign = second
    if sp.simplify(first_base - second_base) != sp.Integer(0):
        return None
    match (first_sign, second_sign):
        case (1, 1):
            combined = sp.Mul(first_argument, second_argument)
        case (1, -1):
            combined = sp.Mul(first_argument, sp.Pow(second_argument, -1, evaluate=False))
        case (-1, 1):
            combined = sp.Mul(second_argument, sp.Pow(first_argument, -1, evaluate=False))
        case _:
            return None
    return first_base, combined


def _exponential_form(base: sp.Basic, exponent: sp.Basic) -> sp.Basic:
    if base is sp.E:
        return sp.exp(exponent)
    return sp.Pow(base, exponent, evaluate=False)


def _single_log_side(equation: sp.Equality) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
    """Return ``(base, argument, other_side)`` when one side is a single logarithm."""
    for target, other in ((equation.lhs, equation.rhs), (equation.rhs, equation.lhs)):
        if other.has(sp.log):
            continue
        parts = _log_parts(target)
        if parts is None:
            continue
        base, argument = parts
        return base, argument, other
    return None


def _log_terms(equation: sp.Equality) -> tuple[sp.Basic, ...]:
    return tuple(
        log_term
        for log_term in sorted(equation.atoms(sp.log), key=str)
        if log_term.args[0].free_symbols
    )


def _append_solved_candidates(
    steps: list[BackendDerivationStep],
    transformed: sp.Equality,
    variable: sp.Symbol,
    candidates: tuple[sp.Basic, ...],
) -> object:
    """Solve a transformed equation and return the expression shown before validation."""
    polynomial_steps = derive_polynomial_equation(transformed, variable, candidates)
    if polynomial_steps:
        steps.extend(polynomial_steps)
        return polynomial_steps[-1].after
    relations = tuple(sp.Eq(variable, candidate, evaluate=False) for candidate in candidates)
    steps.append(
        BackendDerivationStep(
            rule="Solve the resulting equation",
            before=transformed,
            after=relations,
            explanation="Solve the transformed equation exactly for the target variable.",
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The displayed values solve the transformed equation exactly.",
        ),
    )
    return relations


def _append_domain_check(
    steps: list[BackendDerivationStep],
    before: object,
    equation: sp.Equality,
    variable: sp.Symbol,
    candidates: tuple[sp.Basic, ...],
    valid: tuple[sp.Basic, ...],
) -> None:
    checks = tuple(
        BackendMathNote(
            label=(
                f"Test {variable} = {candidate} "
                f"({'satisfies' if candidate in valid else 'rejected'})"
            ),
            expression=BackendIdentity(
                left=equation.lhs.subs(variable, candidate),
                right=equation.rhs.subs(variable, candidate),
            ),
        )
        for candidate in candidates
    )
    steps.append(
        BackendDerivationStep(
            rule="Check candidates in the logarithm's domain",
            before=cast("sp.Basic", before),
            after=tuple(sp.Eq(variable, candidate, evaluate=False) for candidate in valid),
            explanation=(
                "Substitute every candidate into the original equation. Keep only values that "
                "make both sides equal and keep every logarithm's argument positive."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Every remaining candidate satisfies the original equation and its domain."
            ),
            notes=checks,
        )
    )


def _derive_from_log(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
    base: sp.Basic,
    argument: sp.Basic,
    other: sp.Basic,
    *,
    original: sp.Equality | None = None,
    prefix_steps: tuple[BackendDerivationStep, ...] = (),
) -> tuple[BackendDerivationStep, ...]:
    """Exponentiate a single logarithm, solve, then enforce the original domain."""
    domain_equation = equation if original is None else original
    transformed = sp.Eq(argument, _exponential_form(base, other), evaluate=False)
    steps = list(prefix_steps)
    condition = _log_domain_constraints(_log_terms(domain_equation))
    steps.append(
        BackendDerivationStep(
            rule="Exponentiate both sides",
            before=equation,
            after=transformed,
            explanation=(
                "Use the logarithm as the inverse of the exponential. This is valid only where "
                "the logarithm's argument is positive."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail=(
                "Exponentiating a logarithm preserves equality on the logarithm's domain."
            ),
            introduced_constraints=condition,
        )
    )
    candidates = _solve_candidates(transformed, variable)
    if candidates is None:
        return ()
    before = _append_solved_candidates(steps, transformed, variable, candidates)
    valid = tuple(
        candidate
        for candidate in candidates
        if _satisfies_original(domain_equation, variable, candidate)
    )
    if not _roots_match(valid, roots):
        return ()
    _append_domain_check(steps, before, domain_equation, variable, candidates, valid)
    return tuple(steps)


def _derive_single_log(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    single = _single_log_side(equation)
    if single is None:
        return ()
    base, argument, other = single
    return _derive_from_log(equation, variable, roots, base, argument, other)


def _derive_log_sum(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    sum_side: sp.Basic
    other: sp.Basic
    if equation.rhs.has(sp.log):
        if equation.lhs.has(sp.log):
            return ()
        sum_side, other = equation.rhs, equation.lhs
    else:
        sum_side, other = equation.lhs, equation.rhs
    if other.has(variable):
        return ()
    split = _split_log_sum(sum_side)
    if split is None:
        return ()
    base, combined = split
    combined_log = sp.log(combined, base) if base is not sp.E else sp.log(combined)
    combined_equation = sp.Eq(combined_log, other, evaluate=False)
    condition = _log_domain_constraints(_log_terms(equation))
    combine_step = BackendDerivationStep(
        rule="Combine the logarithms",
        before=equation,
        after=combined_equation,
        explanation=(
            "Use the product or quotient rule for logarithms. Both arguments must be positive "
            "before the logarithms are combined."
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "The logarithm laws preserve equality only on the common positive domain of the "
            "original logarithms."
        ),
        notes=(
            BackendMathNote(
                label="Logarithm law",
                expression=BackendIdentity(
                    left=sp.Add(*sum_side.as_ordered_terms(), evaluate=False),
                    right=combined_log,
                ),
            ),
        ),
        introduced_constraints=condition,
    )
    return _derive_from_log(
        combined_equation,
        variable,
        roots,
        base,
        combined,
        other,
        original=equation,
        prefix_steps=(combine_step,),
    )


def _derive_equal_logs(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    left_parts = _log_parts(equation.lhs)
    right_parts = _log_parts(equation.rhs)
    if left_parts is None or right_parts is None:
        return ()
    left_base, left_argument = left_parts
    right_base, right_argument = right_parts
    if sp.simplify(left_base - right_base) != sp.Integer(0):
        return ()
    transformed = sp.Eq(left_argument, right_argument, evaluate=False)
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Equate the logarithms' arguments",
            before=equation,
            after=transformed,
            explanation=(
                "A logarithm is one-to-one on its domain, so equal logarithms imply equal "
                "arguments."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail=(
                "The implication is reversible where both original arguments are positive."
            ),
            introduced_constraints=_log_domain_constraints(_log_terms(equation)),
        )
    ]
    candidates = _solve_candidates(transformed, variable)
    if candidates is None:
        return ()
    before = _append_solved_candidates(steps, transformed, variable, candidates)
    valid = tuple(
        candidate for candidate in candidates if _satisfies_original(equation, variable, candidate)
    )
    if not _roots_match(valid, roots):
        return ()
    _append_domain_check(steps, before, equation, variable, candidates, valid)
    return tuple(steps)


def _constant_and_log(
    expression: sp.Basic,
    variable: sp.Symbol,
) -> tuple[tuple[sp.Basic, sp.Basic], sp.Basic] | None:
    """Split ``log(A, b) + c`` into the logarithm parts and the constant."""
    terms = expression.as_ordered_terms()
    if len(terms) != 2:
        return None
    logarithm_parts: list[tuple[sp.Basic, sp.Basic]] = []
    constants: list[sp.Basic] = []
    for term in terms:
        parts = _log_parts(term)
        if parts is not None:
            logarithm_parts.append(parts)
        elif not term.has(variable):
            constants.append(term)
        else:
            return None
    if len(logarithm_parts) != 1 or len(constants) != 1:
        return None
    return logarithm_parts[0], constants[0]


def _log_with_constant_parts(
    equation: sp.Equality,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic, sp.Basic, sp.Basic] | None:
    """Return ``(base, single argument, collected argument, constant)`` when present."""
    for target, other in ((equation.lhs, equation.rhs), (equation.rhs, equation.lhs)):
        target_parts = _log_parts(target)
        if target_parts is None:
            continue
        other_parts = _constant_and_log(other, variable)
        if other_parts is None:
            continue
        (other_base, other_argument), constant = other_parts
        target_base, target_argument = target_parts
        if sp.simplify(target_base - other_base) != sp.Integer(0):
            continue
        return target_base, target_argument, other_argument, constant
    return None


def _derive_log_sum_with_constant(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Collect two logarithms on one side before exponentiating a constant offset."""
    parts = _log_with_constant_parts(equation, variable)
    if parts is None:
        return ()
    base, target_argument, other_argument, constant = parts
    target_log = sp.log(target_argument, base)
    other_log = sp.log(other_argument, base)
    isolated_equation = sp.Eq(target_log - other_log, constant, evaluate=False)
    isolate_step = BackendDerivationStep(
        rule="Move the logarithms to one side",
        before=equation,
        after=isolated_equation,
        explanation=(
            "Collect both logarithmic terms on the same side so the logarithmic laws can be "
            "applied. Each argument must stay positive."
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "Adding or subtracting the same expression on both sides preserves the solution "
            "set on the logarithms' common domain."
        ),
        introduced_constraints=_log_domain_constraints(_log_terms(equation)),
    )
    combined_argument = sp.Mul(
        target_argument,
        sp.Pow(other_argument, -1, evaluate=False),
        evaluate=False,
    )
    combined_log = sp.log(combined_argument, base)
    combined_equation = sp.Eq(combined_log, constant, evaluate=False)
    combine_step = BackendDerivationStep(
        rule="Combine the logarithms",
        before=isolated_equation,
        after=combined_equation,
        explanation=(
            "The difference of two logarithms with the same base is the logarithm of their "
            "quotient."
        ),
        verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
        verification_detail=(
            "The quotient rule for logarithms is exact where both original arguments are positive."
        ),
        notes=(
            BackendMathNote(
                label="Logarithm law",
                expression=BackendIdentity(
                    left=sp.Add(target_log, -other_log, evaluate=False),
                    right=combined_log,
                ),
            ),
        ),
    )
    return _derive_from_log(
        combined_equation,
        variable,
        roots,
        base,
        combined_argument,
        constant,
        original=equation,
        prefix_steps=(isolate_step, combine_step),
    )


def _derive_log_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    for strategy in (
        _derive_log_sum,
        _derive_log_sum_with_constant,
        _derive_equal_logs,
        _derive_single_log,
    ):
        derivation = strategy(equation, variable, roots)
        if derivation:
            return derivation
    return ()


def _single_exponential_side(
    equation: sp.Equality,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic, sp.Basic] | None:
    """Return ``(base, exponent, other_side)`` for a single exponential side."""
    for target, other in ((equation.lhs, equation.rhs), (equation.rhs, equation.lhs)):
        if other.has(variable):
            continue
        if target.func == sp.exp and len(target.args) == 1:
            return sp.E, target.args[0], other
        if (
            target.is_Pow
            and len(target.args) == 2
            and not target.args[0].has(variable)
            and target.args[1].has(variable)
        ):
            base, exponent = target.args
            return base, exponent, other
    return None


def _same_base_value(base: sp.Basic, value: sp.Basic) -> sp.Basic | None:
    """Return ``c`` with ``base**c = value`` when the representation is exact."""
    try:
        candidate = sp.log(value, base)
    except (TypeError, ValueError):
        return None
    if sp.simplify(sp.Pow(base, candidate, evaluate=False) - value) == sp.Integer(0):
        return sp.simplify(candidate)
    return None


def _derive_single_exponential(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    single = _single_exponential_side(equation, variable)
    if single is None:
        return ()
    base, exponent, other = single
    same_base = None if base is sp.E else _same_base_value(base, other)
    if same_base is not None:
        transformed = sp.Eq(exponent, same_base, evaluate=False)
        rule = "Write both sides with the same base"
        explanation = (
            "Express the constant on the right as a power of the same base, then equate the "
            "exponents."
        )
        notes: tuple[BackendMathNote, ...] = (
            BackendMathNote(
                label="Matching power",
                expression=BackendIdentity(
                    left=_exponential_form(base, same_base),
                    right=other,
                ),
            ),
        )
        constraints: tuple[BackendStepConstraint, ...] = ()
    else:
        logarithm = sp.log(other) if base is sp.E else sp.log(other, base)
        transformed = sp.Eq(exponent, logarithm, evaluate=False)
        rule = "Take logarithms of both sides"
        explanation = (
            "Apply the same logarithm to both sides. This requires the exponential side to be "
            "positive."
        )
        notes = ()
        constraints = (
            BackendStepConstraint(
                explanation="An exponential is positive, so the other side must be positive.",
                expression=sp.Gt(other, sp.Integer(0), evaluate=False),
            ),
        )
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule=rule,
            before=equation,
            after=transformed,
            explanation=explanation,
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The exponential is one-to-one on its range.",
            notes=notes,
            introduced_constraints=constraints,
        )
    ]
    candidates = _solve_candidates(transformed, variable)
    if candidates is None:
        return ()
    before = _append_solved_candidates(steps, transformed, variable, candidates)
    valid = tuple(
        candidate for candidate in candidates if _satisfies_original(equation, variable, candidate)
    )
    if not _roots_match(valid, roots):
        return ()
    checks = tuple(
        BackendMathNote(
            label=(
                f"Test {variable} = {candidate} "
                f"({'satisfies' if candidate in valid else 'rejected'})"
            ),
            expression=BackendIdentity(
                left=equation.lhs.subs(variable, candidate),
                right=equation.rhs.subs(variable, candidate),
            ),
        )
        for candidate in candidates
    )
    steps.append(
        BackendDerivationStep(
            rule="Check candidates in the original equation",
            before=cast("sp.Basic", before),
            after=tuple(sp.Eq(variable, candidate, evaluate=False) for candidate in valid),
            explanation="Substitute every candidate back into the original exponential equation.",
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Every remaining candidate satisfies the original equation.",
            notes=checks,
        )
    )
    return tuple(steps)


def _derive_equal_exponentials(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    if equation.lhs.func != sp.exp or equation.rhs.func != sp.exp:
        return ()
    left_argument = equation.lhs.args[0]
    right_argument = equation.rhs.args[0]
    transformed = sp.Eq(left_argument, right_argument, evaluate=False)
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule="Equate the exponents",
            before=equation,
            after=transformed,
            explanation=(
                "The natural exponential is one-to-one, so equal exponentials have equal exponents."
            ),
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="Exponentiating equal exponents recovers the original equation.",
        )
    ]
    candidates = _solve_candidates(transformed, variable)
    if candidates is None:
        return ()
    before = _append_solved_candidates(steps, transformed, variable, candidates)
    valid = tuple(
        candidate for candidate in candidates if _satisfies_original(equation, variable, candidate)
    )
    if not _roots_match(valid, roots):
        return ()
    _append_domain_check(steps, before, equation, variable, candidates, valid)
    return tuple(steps)


def _positive_slopes(
    exp_atoms: tuple[sp.Basic, ...],
    variable: sp.Symbol,
) -> tuple[sp.Basic, tuple[tuple[sp.Basic, sp.Basic], ...]] | None:
    """Return a common positive slope for an exponential substitution."""
    slopes: list[sp.Basic] = []
    slopes_by_atom: list[tuple[sp.Basic, sp.Basic]] = []
    for atom in exp_atoms:
        argument = atom.args[0]
        slope = sp.diff(argument, variable)
        constant = sp.simplify(argument - slope * variable)
        if (
            slope.has(variable)
            or constant != sp.Integer(0)
            or slope.is_rational is not True
            or slope.is_positive is not True
        ):
            return None
        slopes.append(slope)
        slopes_by_atom.append((atom, slope))
    base_slope = min(slopes)
    if not all((slope / base_slope).is_integer for slope in slopes):
        return None
    return base_slope, tuple(slopes_by_atom)


def _prime_exponents(base: sp.Basic) -> dict[sp.Integer, int]:
    """Return the prime factorization of a positive rational base as a dictionary."""
    rational = sp.Rational(cast("int", base))
    exponents: dict[sp.Integer, int] = {}
    for prime, exponent in sp.factorint(rational.p).items():
        key = sp.Integer(prime)
        exponents[key] = exponents.get(key, 0) + int(exponent)
    for prime, exponent in sp.factorint(rational.q).items():
        key = sp.Integer(prime)
        exponents[key] = exponents.get(key, 0) - int(exponent)
    return {prime: exponent for prime, exponent in exponents.items() if exponent}


def _common_power_base(
    factorizations: tuple[dict[sp.Integer, int], ...],
) -> tuple[sp.Basic, tuple[int, ...]] | None:
    """Find a common base when every rational base is a power of it."""
    if not factorizations:
        return None
    weights: list[int] = []
    normalized_bases: list[dict[sp.Integer, int]] = []
    for factorization in factorizations:
        if not factorization:
            return None
        divisors = [abs(exponent) for exponent in factorization.values()]
        common_divisor = 0
        for divisor in divisors:
            common_divisor = gcd(common_divisor, divisor)
        if common_divisor == 0:
            return None
        normalized_bases.append(
            {prime: exponent // common_divisor for prime, exponent in factorization.items()}
        )
        weights.append(common_divisor)
    unit = normalized_bases[0]
    if any(normalized != unit for normalized in normalized_bases[1:]):
        return None
    common: sp.Basic = sp.Integer(1)
    for prime, exponent in unit.items():
        common *= prime**exponent
    return common, tuple(weights)


def _detect_power_substitution(
    equation: sp.Equality,
    variable: sp.Symbol,
) -> tuple[sp.Basic, sp.Basic, tuple[tuple[sp.Basic, sp.Basic], ...]] | None:
    """Express variable-base powers through one common exponential substitution."""
    powers: list[tuple[sp.Basic, sp.Basic, sp.Basic]] = []
    for atom in equation.atoms(sp.Pow):
        base, exponent = atom.args
        if base.is_rational is not True or base.is_positive is not True or base == sp.Integer(1):
            continue
        slope = sp.diff(exponent, variable)
        if (
            slope.has(variable)
            or slope == sp.Integer(0)
            or sp.simplify(exponent - slope * variable) != sp.Integer(0)
        ):
            continue
        powers.append((atom, base, slope))
    if not powers:
        return None
    detection = _common_power_base(tuple(_prime_exponents(base) for _, base, _ in powers))
    if detection is None:
        return None
    common_base, weights = detection
    try:
        effective = tuple(
            weight * sp.Rational(cast("int", slope))
            for weight, (_, _, slope) in zip(weights, powers, strict=True)
        )
    except (TypeError, ValueError):
        return None
    if not all(value.is_rational is True and value.is_positive is True for value in effective):
        return None
    base_slope: sp.Basic = effective[0]
    for value in effective[1:]:
        base_slope = sp.gcd(base_slope, value)
    powers_by_atom: list[tuple[sp.Basic, sp.Basic]] = []
    for (atom, _, _), value in zip(powers, effective, strict=True):
        power: sp.Basic = sp.simplify(value / base_slope)
        powers_by_atom.append((atom, power))
    return common_base, base_slope, tuple(powers_by_atom)


def _derive_exponential_substitution(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    exp_atoms = tuple(sorted(equation.atoms(sp.exp), key=str))
    substitution_base: sp.Basic | None = None
    base_slope: sp.Basic
    powers_by_atom: tuple[tuple[sp.Basic, sp.Basic], ...]
    if exp_atoms:
        detection = _positive_slopes(exp_atoms, variable)
        if detection is None:
            return ()
        base_slope, slope_pairs = detection
        powers_by_atom = tuple(
            (atom, sp.simplify(slope / base_slope)) for atom, slope in slope_pairs
        )
    else:
        power_detection = _detect_power_substitution(equation, variable)
        if power_detection is None:
            return ()
        substitution_base, base_slope, powers_by_atom = power_detection
    substitution_expression: sp.Basic
    if substitution_base is None:
        substitution_expression = sp.exp(base_slope * variable)
        logarithm_base: sp.Basic = sp.E
        substitution_display = f"exp({_SSTR(sp.simplify(base_slope * variable))})"
    else:
        substitution_expression = sp.Pow(substitution_base, base_slope * variable, evaluate=False)
        logarithm_base = substitution_base
        substitution_display = (
            f"{_SSTR(substitution_base)}^{_SSTR(sp.simplify(base_slope * variable))}"
        )
    substitution = sp.Symbol("u", real=True)
    difference = equation.lhs - equation.rhs
    replaced: sp.Basic = difference
    replacements = {atom: substitution**power for atom, power in powers_by_atom}
    replaced = replaced.xreplace(replacements)
    replaced = sp.expand(replaced)
    if replaced.has(variable) or not replaced.is_polynomial(substitution):
        return ()
    polynomial = sp.Poly(replaced, substitution)
    if polynomial.degree() < _QUADRATIC_DEGREE:
        return ()
    u_roots = _solve_candidates(sp.Eq(replaced, sp.Integer(0), evaluate=False), substitution)
    if u_roots is None:
        return ()
    positive_roots = tuple(root for root in u_roots if sp.simplify(root).is_positive is True)
    if len(positive_roots) != len(u_roots) and any(
        sp.simplify(root).is_positive is None for root in u_roots
    ):
        return ()
    final_roots = tuple(
        sp.simplify(sp.log(root) / (base_slope * sp.log(logarithm_base))) for root in positive_roots
    )
    if not _roots_match(final_roots, roots):
        return ()
    if not all(_satisfies_original(equation, variable, candidate) for candidate in final_roots):
        return ()
    substitution_equation = sp.Eq(replaced, sp.Integer(0), evaluate=False)
    u_relations = tuple(sp.Eq(substitution, root, evaluate=False) for root in u_roots)
    positive_relations = tuple(sp.Eq(substitution, root, evaluate=False) for root in positive_roots)
    final_relations = tuple(sp.Eq(variable, candidate, evaluate=False) for candidate in final_roots)
    steps: list[BackendDerivationStep] = [
        BackendDerivationStep(
            rule=f"Substitute u = {substitution_display}",
            before=equation,
            after=substitution_equation,
            explanation=(
                "The equation is a polynomial in the exponential term. Introduce a positive "
                "substitution so the quadratic can be solved directly."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail=(
                "Replacing the exponential by u is reversible on the positive domain u > 0."
            ),
            notes=(
                BackendMathNote(
                    label="Substitution",
                    expression=sp.Eq(
                        substitution,
                        substitution_expression,
                        evaluate=False,
                    ),
                ),
            ),
            introduced_constraints=(
                BackendStepConstraint(
                    explanation="The exponential substitution is positive.",
                    expression=sp.Gt(substitution, sp.Integer(0), evaluate=False),
                ),
            ),
        ),
        BackendDerivationStep(
            rule="Solve the polynomial in u",
            before=substitution_equation,
            after=u_relations,
            explanation="Factor or apply the quadratic formula to the polynomial in u.",
            verification_method=VerificationMethod.SOLUTION_SET_EQUIVALENCE,
            verification_detail="The displayed values satisfy the polynomial exactly.",
        ),
    ]
    if len(positive_roots) != len(u_roots):
        steps.append(
            BackendDerivationStep(
                rule="Reject nonpositive values of u",
                before=u_relations,
                after=positive_relations,
                explanation=(
                    f"Since u = {substitution_display} is always positive, any zero or "
                    "negative root of the polynomial is impossible."
                ),
                verification_method=VerificationMethod.EXACT_ARITHMETIC,
                verification_detail="An exponential can never equal a nonpositive real number.",
            )
        )
    steps.append(
        BackendDerivationStep(
            rule="Recover the variable from the substitution",
            before=positive_relations,
            after=final_relations,
            explanation=(
                "Take the logarithm base "
                f"{_SSTR(logarithm_base)} of each positive value of u, then divide by the "
                f"exponential rate {_SSTR(base_slope)}."
            ),
            verification_method=VerificationMethod.SUBSTITUTION,
            verification_detail="Each recovered value satisfies the original equation.",
        )
    )
    return tuple(steps)


def _has_variable_power(expression: sp.Basic, variable: sp.Symbol) -> bool:
    return any(
        power.args[0].has(variable) is False and power.args[1].has(variable)
        for power in expression.atoms(sp.Pow)
    )


def derive_exponential_logarithmic_equation(
    equation: sp.Equality,
    variable: sp.Symbol,
    roots: tuple[sp.Basic, ...],
) -> tuple[BackendDerivationStep, ...]:
    """Derive logarithmic and exponential equations from their structure."""
    if equation.has(sp.log):
        derivation = _derive_log_equation(equation, variable, roots)
        if derivation:
            return derivation
    if equation.has(sp.exp) or _has_variable_power(equation, variable):
        for strategy in (
            _derive_equal_exponentials,
            _derive_single_exponential,
            _derive_exponential_substitution,
        ):
            derivation = strategy(equation, variable, roots)
            if derivation:
                return derivation
    return ()
