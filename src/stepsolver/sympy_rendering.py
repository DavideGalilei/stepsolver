"""Translate backend derivations into the public step model."""

import sympy as sp

from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    FunctionCall,
    Identifier,
    Operation,
    Relation,
    RelationOperator,
    SequenceExpression,
    Symbol,
)
from stepsolver.derivation.model import (
    BackendApproximateSolutions,
    BackendCardanoSolution,
    BackendCrossedOut,
    BackendDerivationStep,
    BackendDerivative,
    BackendDifference,
    BackendDifferential,
    BackendEvaluationAtBounds,
    BackendEvaluationAtIndex,
    BackendExpression,
    BackendGrouped,
    BackendIdentity,
    BackendIntegral,
    BackendIntegrationByPartsRule,
    BackendIntroducedOperation,
    BackendIntroducedProduct,
    BackendIntroducedQuotient,
    BackendLimit,
    BackendNewtonIterations,
    BackendNewtonRule,
    BackendNotEqual,
    BackendNthRoot,
    BackendProduct,
    BackendQuadraticSolutions,
    BackendQuotient,
    BackendRowOperation,
    BackendSigma,
    BackendSum,
    BackendSystem,
    BackendUndefined,
)
from stepsolver.results import (
    InlineMath,
    SolutionStep,
    StepConstraint,
    StepNote,
    Verification,
)
from stepsolver.sympy_conversion import SympyConverter


class SympyDerivationRenderer:
    """Render backend-native derivation objects as StepSolver expressions and steps."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the supplied scalar converter inside displayed derivations."""
        self._converter = converter

    def solution_step(self, step: BackendDerivationStep) -> SolutionStep:
        """Convert one backend derivation step to the public step model."""
        return SolutionStep(
            rule=step.rule,
            before=self.derivation_expression(step.before),
            after=self.derivation_expression(step.after),
            explanation=step.explanation,
            explanation_parts=tuple(
                part
                if isinstance(part, str)
                else InlineMath(expression=self.derivation_expression(part.expression))
                for part in step.explanation_parts
            ),
            verification=Verification(
                method=step.verification_method,
                detail=step.verification_detail,
            ),
            notes=tuple(
                StepNote(
                    label=note.label,
                    expression=self.derivation_expression(note.expression),
                )
                for note in step.notes
            ),
            introduced_constraints=tuple(
                StepConstraint(
                    explanation=constraint.explanation,
                    expression=self.derivation_expression(constraint.expression),
                )
                for constraint in step.introduced_constraints
            ),
        )

    def derivation_expression(
        self,
        value: BackendExpression,
    ) -> Expression:
        """Convert one backend display object to the public mathematical AST."""
        if isinstance(value, BackendIdentity):
            return Relation(
                operator=RelationOperator.EQUAL,
                left=self.derivation_expression(value.left),
                right=self.derivation_expression(value.right),
            )
        if isinstance(value, BackendIntegrationByPartsRule):
            return FunctionCall(name=Identifier("integration_by_parts_rule"), arguments=())
        if isinstance(value, BackendQuadraticSolutions):
            return FunctionCall(
                name=Identifier("quadratic_solutions"),
                arguments=(
                    self._converter.from_sympy(value.variable),
                    self.derivation_expression(value.negative_numerator),
                    self.derivation_expression(value.positive_numerator),
                    self.derivation_expression(value.denominator),
                ),
            )
        if isinstance(value, BackendCardanoSolution):
            return FunctionCall(
                name=Identifier("cardano_solution"),
                arguments=(
                    self._converter.from_sympy(value.variable),
                    self._converter.from_sympy(value.shift),
                    self._converter.from_sympy(value.first_radicand),
                    self._converter.from_sympy(value.second_radicand),
                ),
            )
        if isinstance(value, BackendCrossedOut):
            return FunctionCall(
                name=Identifier("crossed_out"),
                arguments=(self.derivation_expression(value.expression),),
            )
        if isinstance(value, BackendIntroducedProduct):
            return FunctionCall(
                name=Identifier("introduced_product"),
                arguments=(
                    self.derivation_expression(value.multiplier),
                    self.derivation_expression(value.expression),
                ),
            )
        if isinstance(value, BackendIntroducedOperation):
            return FunctionCall(
                name=Identifier(f"introduced_{value.operation}"),
                arguments=(
                    self.derivation_expression(value.expression),
                    self.derivation_expression(value.operand),
                ),
            )
        if isinstance(value, BackendIntroducedQuotient):
            return FunctionCall(
                name=Identifier("introduced_quotient"),
                arguments=(
                    self.derivation_expression(value.numerator),
                    self.derivation_expression(value.denominator),
                ),
            )
        if isinstance(value, BackendGrouped):
            return FunctionCall(
                name=Identifier("grouped"),
                arguments=(self.derivation_expression(value.expression),),
            )
        if isinstance(value, BackendNthRoot):
            return FunctionCall(
                name=Identifier("nth_root"),
                arguments=(
                    self.derivation_expression(value.radicand),
                    self.derivation_expression(value.index),
                ),
            )
        if isinstance(value, BackendNewtonRule):
            return FunctionCall(name=Identifier("newton_rule"), arguments=())
        if isinstance(value, BackendNewtonIterations):
            return FunctionCall(
                name=Identifier("newton_iterations"),
                arguments=(
                    self._converter.from_sympy(value.variable),
                    *(self._converter.from_sympy(item) for item in value.values),
                ),
            )
        if isinstance(value, BackendApproximateSolutions):
            return FunctionCall(
                name=Identifier("approximate_solutions"),
                arguments=(
                    self._converter.from_sympy(value.variable),
                    *(self._converter.from_sympy(root) for root in value.roots),
                ),
            )
        if isinstance(value, BackendSigma):
            return FunctionCall(
                name=Identifier(Operation.SUM.value),
                arguments=(
                    self.derivation_expression(value.expression),
                    self._converter.from_sympy(value.variable),
                    self._converter.from_sympy(value.lower),
                    self._converter.from_sympy(value.upper),
                ),
            )
        if isinstance(value, BackendEvaluationAtIndex):
            return FunctionCall(
                name=Identifier("evaluate_at_index"),
                arguments=(
                    self.derivation_expression(value.expression),
                    self._converter.from_sympy(value.variable),
                    self._converter.from_sympy(value.index),
                ),
            )
        if isinstance(value, BackendUndefined):
            return FunctionCall(name=Identifier("undefined"), arguments=())
        if isinstance(value, BackendNotEqual):
            return Relation(
                operator=RelationOperator.NOT_EQUAL,
                left=self.derivation_expression(value.left),
                right=self.derivation_expression(value.right),
            )
        if isinstance(value, BackendSystem):
            return FunctionCall(
                name=Identifier("system"),
                arguments=tuple(self._converter.from_sympy(item) for item in value.equations),
            )
        if isinstance(value, BackendRowOperation):
            return FunctionCall(
                name=Identifier("row_operation"),
                arguments=(
                    self._converter.from_sympy(sp.Integer(value.target)),
                    self._converter.from_sympy(sp.Integer(value.source)),
                    self._converter.from_sympy(value.factor),
                ),
            )
        if isinstance(value, BackendSum):
            expressions = tuple(self.derivation_expression(term) for term in value.terms)
            first, *remaining = expressions
            result = first
            for expression in remaining:
                result = BinaryExpression(
                    operator=BinaryOperator.ADD,
                    left=result,
                    right=expression,
                )
            return result
        if isinstance(value, BackendProduct):
            expressions = tuple(self.derivation_expression(factor) for factor in value.factors)
            first, *remaining = expressions
            result = first
            for expression in remaining:
                result = BinaryExpression(
                    operator=BinaryOperator.MULTIPLY,
                    left=result,
                    right=expression,
                )
            return result
        if isinstance(value, BackendQuotient):
            return BinaryExpression(
                operator=BinaryOperator.DIVIDE,
                left=self.derivation_expression(value.numerator),
                right=self.derivation_expression(value.denominator),
            )
        if isinstance(value, BackendDifference):
            return BinaryExpression(
                operator=BinaryOperator.SUBTRACT,
                left=self.derivation_expression(value.left),
                right=self.derivation_expression(value.right),
            )
        if isinstance(value, BackendDifferential):
            differential = FunctionCall(
                name=Identifier("differential"),
                arguments=(self._converter.from_sympy(value.variable),),
            )
            if value.coefficient is None:
                return differential
            return BinaryExpression(
                operator=BinaryOperator.MULTIPLY,
                left=self._converter.from_sympy(value.coefficient),
                right=differential,
            )
        if isinstance(value, BackendDerivative):
            return FunctionCall(
                name=Identifier(Operation.DIFFERENTIATE.value),
                arguments=(
                    self._converter.from_sympy(value.expression),
                    self._converter.from_sympy(value.variable),
                ),
            )
        if isinstance(value, BackendEvaluationAtBounds):
            return FunctionCall(
                name=Identifier("evaluate_at_bounds"),
                arguments=(
                    self._converter.from_sympy(value.expression),
                    self._converter.from_sympy(value.variable),
                    self._converter.from_sympy(value.lower),
                    self._converter.from_sympy(value.upper),
                ),
            )
        if isinstance(value, BackendLimit):
            limit_arguments: tuple[Expression, ...] = (
                self.derivation_expression(value.expression),
                self._converter.from_sympy(value.variable),
                self._converter.from_sympy(value.point),
            )
            if value.direction is not None:
                direction = "right" if value.direction == "+" else "left"
                limit_arguments = (*limit_arguments, Symbol(name=Identifier(direction)))
            return FunctionCall(
                name=Identifier(Operation.LIMIT.value),
                arguments=limit_arguments,
            )
        if isinstance(value, BackendIntegral):
            integral_arguments: tuple[Expression, ...] = (
                self._converter.from_sympy(value.integrand),
                self._converter.from_sympy(value.variable),
            )
            if value.lower is not None and value.upper is not None:
                integral_arguments = (
                    *integral_arguments,
                    self._converter.from_sympy(value.lower),
                    self._converter.from_sympy(value.upper),
                )
            integral = FunctionCall(
                name=Identifier(Operation.INTEGRATE.value),
                arguments=integral_arguments,
            )
            if value.coefficient is None:
                return integral
            return BinaryExpression(
                operator=BinaryOperator.MULTIPLY,
                left=self._converter.from_sympy(value.coefficient),
                right=integral,
            )
        if isinstance(value, tuple):
            return SequenceExpression(
                items=tuple(self._converter.from_sympy(item) for item in value)
            )
        return self._converter.from_sympy(value)
