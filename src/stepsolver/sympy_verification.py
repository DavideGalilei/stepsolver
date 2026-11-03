"""Independent verification of backend results."""

import sympy as sp

from stepsolver.ast import (
    Operation,
    Query,
)
from stepsolver.errors import BackendError
from stepsolver.results import (
    Verification,
    VerificationMethod,
)
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_equivalence import expressions_equivalent
from stepsolver.sympy_support import (
    expect_integer,
    negative_condition,
    positive_condition,
    refine_expression,
)


class SympyVerifier:
    """Verify exact backend results against the original query."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the supplied converter to reconstruct expected backend values."""
        self._converter = converter

    def verify_result(self, query: Query, backend_value: object) -> Verification:
        """Verify a backend result using an operation-appropriate method."""
        operation = query.operation
        if operation in {
            Operation.SIMPLIFY,
            Operation.EXPAND,
            Operation.FACTOR,
            Operation.CANCEL,
            Operation.APART,
        }:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("algebraic verification requires a scalar result")
            original = self._converter.to_sympy(query.arguments[0])
            if not expressions_equivalent(original, backend_value):
                raise BackendError("the algebraic result failed symbolic equivalence verification")
            return Verification(
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="Simplifying the difference between both expressions produced zero.",
            )
        if operation is Operation.DIFFERENTIATE:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("derivative verification requires a scalar result")
            order = (
                expect_integer(query.arguments[2], role="derivative order")
                if len(query.arguments) == 3
                else 1
            )
            expected = sp.diff(
                self._converter.to_sympy(query.arguments[0]),
                self._converter.to_sympy(query.arguments[1]),
                order,
            )
            if not expressions_equivalent(expected, backend_value):
                raise BackendError("the derivative result failed symbolic verification")
            return Verification(
                method=VerificationMethod.DIFFERENTIATION,
                detail="Recomputing the derivative and simplifying the difference produced zero.",
            )
        if operation is Operation.INTEGRATE and len(query.arguments) == 2:
            if not isinstance(backend_value, sp.Basic):
                raise BackendError("antiderivative verification requires a scalar result")
            variable = self._converter.to_sympy(query.arguments[1])
            integrand = self._converter.to_sympy(query.arguments[0])
            derivative = sp.diff(backend_value, variable)
            difference = derivative - integrand
            if not expressions_equivalent(derivative, integrand):
                if not (
                    isinstance(variable, sp.Symbol)
                    and variable.is_real is True
                    and self._differentiates_away_from_real_poles(
                        difference,
                        integrand,
                        backend_value,
                        variable,
                    )
                ):
                    raise BackendError("the antiderivative failed differentiation verification")
                return Verification(
                    method=VerificationMethod.DIFFERENTIATION,
                    detail=(
                        "Differentiating the result recovers the integrand on both real "
                        "intervals separated by the excluded value."
                    ),
                )
            return Verification(
                method=VerificationMethod.DIFFERENTIATION,
                detail="Differentiating the result recovered the original integrand.",
            )
        if operation in {
            Operation.GCD,
            Operation.LCM,
            Operation.IS_PRIME,
            Operation.PRIME_FACTORS,
            Operation.BINOMIAL,
            Operation.PERMUTATIONS,
            Operation.COMBINATIONS,
        }:
            return Verification(
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The operation was evaluated using exact integer arithmetic.",
            )
        return Verification(
            method=VerificationMethod.BACKEND_IDENTITY,
            detail="The exact operation completed and crossed the validated typed boundary.",
        )

    @staticmethod
    def _differentiates_away_from_real_poles(
        difference: sp.Basic,
        integrand: sp.Basic,
        backend_value: sp.Basic,
        variable: sp.Symbol,
    ) -> bool:
        """Check an absolute-value antiderivative on both sides of its real poles."""
        if not backend_value.has(sp.Abs):
            return False
        _numerator, denominator = sp.fraction(sp.together(integrand))
        conditions: list[object] = []
        if denominator.has(variable):
            conditions.extend((positive_condition(denominator), negative_condition(denominator)))
        for absolute in backend_value.atoms(sp.Abs):
            if absolute.has(variable):
                conditions.extend((positive_condition(absolute), negative_condition(absolute)))
        return bool(conditions) and all(
            sp.simplify(refine_expression(difference, condition)) == sp.Integer(0)
            for condition in conditions
        )
