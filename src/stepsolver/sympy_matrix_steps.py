"""Student-facing derivations for matrix operations.

Each builder recomputes its displayed intermediate values exactly and compares the final
value with the backend result.  When the comparison fails, or the input is outside the
documented scope, the builder returns no steps instead of presenting an unverified
derivation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import sympy as sp

from stepsolver.ast import (
    BinaryExpression,
    BinaryOperator,
    Expression,
    FunctionCall,
    Identifier,
    Number,
    Operation,
    Query,
    Relation,
    RelationOperator,
    SequenceExpression,
)
from stepsolver.results import SolutionStep, StepNote, Verification, VerificationMethod
from stepsolver.sympy_conversion import SympyConverter
from stepsolver.sympy_support import is_object_mapping, query_expression

_SMALLEST_EXPANDED_SIZE = 2
_EIGENVALUE_SYMBOL = sp.Symbol("lambda")

type MatrixStepFactory = Callable[[Query, sp.Matrix, object], tuple[SolutionStep, ...]]


@dataclass(frozen=True, slots=True)
class _RowReduction:
    """The exact Gauss-Jordan steps for one numeric matrix."""

    steps: tuple[SolutionStep, ...]
    reduced: sp.Matrix
    pivots: tuple[int, ...]


class SympyMatrixStepBuilder:
    """Build verified human steps for the supported matrix operations."""

    def __init__(self, converter: SympyConverter) -> None:
        """Use the shared converter to display backend matrices as public expressions."""
        self._converter = converter

    def steps(self, query: Query, backend_value: object) -> tuple[SolutionStep, ...]:
        """Return steps for a matrix query, or an empty tuple when none can be verified."""
        factories: dict[Operation, MatrixStepFactory] = {
            Operation.MATRIX: self._matrix_steps,
            Operation.DETERMINANT: self._determinant_steps,
            Operation.INVERSE: self._inverse_steps,
            Operation.RANK: self._rank_steps,
            Operation.RREF: self._rref_steps,
            Operation.EIGENVALUES: self._eigenvalue_steps,
        }
        factory = factories.get(query.operation)
        if factory is None or len(query.arguments) != 1:
            return ()
        matrix = cast("sp.Matrix", self._converter.matrix_from_expression(query.arguments[0]))
        return factory(query, matrix, backend_value)

    # Display helpers -----------------------------------------------------------------

    def _matrix(self, matrix: sp.MatrixBase) -> Expression:
        rows = tuple(
            SequenceExpression(
                items=tuple(
                    self._converter.from_sympy(matrix[row, column]) for column in range(matrix.cols)
                )
            )
            for row in range(matrix.rows)
        )
        return FunctionCall(
            name=Identifier("matrix"),
            arguments=(SequenceExpression(items=rows),),
        )

    def _scalar(self, value: sp.Basic) -> Expression:
        return self._converter.from_sympy(value)

    @staticmethod
    def _integer(value: int) -> Number:
        return Number(value=Fraction(value))

    @staticmethod
    def _binary(operator: BinaryOperator, left: Expression, right: Expression) -> Expression:
        return BinaryExpression(operator=operator, left=left, right=right)

    @staticmethod
    def _step(
        *,
        rule: str,
        before: Expression,
        after: Expression,
        explanation: str,
        method: VerificationMethod,
        detail: str,
        notes: tuple[StepNote, ...] = (),
    ) -> SolutionStep:
        return SolutionStep(
            rule=rule,
            before=before,
            after=after,
            explanation=explanation,
            verification=Verification(method=method, detail=detail),
            notes=notes,
        )

    @staticmethod
    def _is_zero(value: sp.Basic) -> bool:
        return sp.simplify(value) == sp.Integer(0)

    def _same_matrix(self, left: sp.MatrixBase, right: sp.MatrixBase) -> bool:
        if left.shape != right.shape:
            return False
        return all(self._is_zero(item) for item in (left - right))

    # Operations ----------------------------------------------------------------------

    def _matrix_steps(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if not isinstance(backend_value, sp.MatrixBase) or not self._same_matrix(
            matrix, backend_value
        ):
            return ()
        shape = SequenceExpression(
            items=(self._integer(matrix.rows), self._integer(matrix.cols)),
        )
        return (
            self._step(
                rule="Read the matrix entries",
                before=query_expression(query),
                after=self._matrix(matrix),
                explanation=(
                    f"Read the {matrix.rows} by {matrix.cols} array as a matrix with one row per "
                    "inner list and one column per entry position."
                ),
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail=f"Every row contains exactly {matrix.cols} entries, so the shape is valid.",
                notes=(StepNote(label="Rows and columns", expression=shape),),
            ),
        )

    def _determinant_steps(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if not isinstance(backend_value, sp.Basic) or not matrix.is_square:
            return ()
        if matrix.rows == 1:
            return self._single_entry_determinant(query, matrix, backend_value)
        if matrix.rows == _SMALLEST_EXPANDED_SIZE:
            return self._two_by_two_determinant(query, matrix, backend_value)
        return self._cofactor_determinant(query, matrix, backend_value)

    def _single_entry_determinant(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: sp.Basic,
    ) -> tuple[SolutionStep, ...]:
        if not self._is_zero(matrix[0, 0] - backend_value):
            return ()
        return (
            self._step(
                rule="Determinant of a 1 by 1 matrix",
                before=query_expression(query),
                after=self._scalar(backend_value),
                explanation="The determinant of a 1 by 1 matrix is its only entry.",
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The only entry equals the backend determinant.",
                notes=(StepNote(label="Matrix", expression=self._matrix(matrix)),),
            ),
        )

    def _two_by_two_determinant(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: sp.Basic,
    ) -> tuple[SolutionStep, ...]:
        a, b, c, d = (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])
        if not self._is_zero(a * d - b * c - backend_value):
            return ()
        diagonal = self._binary(BinaryOperator.MULTIPLY, self._scalar(a), self._scalar(d))
        anti_diagonal = self._binary(BinaryOperator.MULTIPLY, self._scalar(b), self._scalar(c))
        formula = self._binary(BinaryOperator.SUBTRACT, diagonal, anti_diagonal)
        return (
            self._step(
                rule="Apply the 2 by 2 determinant formula",
                before=query_expression(query),
                after=formula,
                explanation=(
                    "For a 2 by 2 matrix, multiply the main diagonal and subtract the product of "
                    "the other diagonal."
                ),
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="The formula a*d - b*c is the determinant of a 2 by 2 matrix.",
                notes=(StepNote(label="Matrix", expression=self._matrix(matrix)),),
            ),
            self._step(
                rule="Evaluate the products and subtract",
                before=formula,
                after=self._scalar(backend_value),
                explanation="Multiply each diagonal pair and subtract the results exactly.",
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The evaluated formula equals the backend determinant.",
                notes=(
                    StepNote(label="Main diagonal product", expression=self._scalar(a * d)),
                    StepNote(label="Other diagonal product", expression=self._scalar(b * c)),
                ),
            ),
        )

    def _cofactor_determinant(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: sp.Basic,
    ) -> tuple[SolutionStep, ...]:
        minors = tuple(matrix.minor_submatrix(0, column) for column in range(matrix.cols))
        minor_values = tuple(minor.det() for minor in minors)
        expansion_value = sp.Add(
            *(
                (-1) ** column * matrix[0, column] * minor_values[column]
                for column in range(matrix.cols)
            )
        )
        if not self._is_zero(expansion_value - backend_value):
            return ()
        expansion = self._first_row_expansion(matrix, minors)
        return (
            self._step(
                rule="Expand along the first row",
                before=query_expression(query),
                after=expansion,
                explanation=(
                    "Multiply each first-row entry by the determinant of the matrix left after "
                    "removing its row and column, alternating signs."
                ),
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="Laplace expansion along a row gives the determinant.",
                notes=tuple(
                    StepNote(label=f"Minor {column + 1}", expression=self._matrix(minor))
                    for column, minor in enumerate(minors)
                ),
            ),
            self._step(
                rule="Evaluate the smaller determinants",
                before=expansion,
                after=self._scalar(backend_value),
                explanation="Evaluate every minor determinant, multiply, and add with the signs.",
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The signed expansion equals the backend determinant.",
                notes=tuple(
                    StepNote(
                        label=f"Determinant of minor {column + 1}", expression=self._scalar(value)
                    )
                    for column, value in enumerate(minor_values)
                ),
            ),
        )

    def _first_row_expansion(
        self,
        matrix: sp.Matrix,
        minors: tuple[sp.Matrix, ...],
    ) -> Expression:
        if not minors:
            return self._scalar(sp.Integer(0))
        first_minor = minors[0]
        expression = self._binary(
            BinaryOperator.MULTIPLY,
            self._scalar(matrix[0, 0]),
            FunctionCall(
                name=Identifier(Operation.DETERMINANT.value),
                arguments=(self._matrix(first_minor),),
            ),
        )
        for column, minor in enumerate(minors[1:], start=1):
            term = self._binary(
                BinaryOperator.MULTIPLY,
                self._scalar(matrix[0, column]),
                FunctionCall(
                    name=Identifier(Operation.DETERMINANT.value), arguments=(self._matrix(minor),)
                ),
            )
            operator = BinaryOperator.ADD if column % 2 == 0 else BinaryOperator.SUBTRACT
            expression = self._binary(operator, expression, term)
        return expression

    def _inverse_steps(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if not isinstance(backend_value, sp.MatrixBase) or not matrix.is_square:
            return ()
        determinant = matrix.det()
        adjugate = matrix.adjugate()
        if self._is_zero(determinant) or not self._same_matrix(
            adjugate / determinant, backend_value
        ):
            return ()
        identity = sp.eye(matrix.rows)
        if not self._same_matrix(matrix * backend_value, identity):
            return ()
        scaled = self._binary(
            BinaryOperator.DIVIDE, self._matrix(adjugate), self._scalar(determinant)
        )
        return (
            self._step(
                rule="Check that the matrix is invertible",
                before=query_expression(query),
                after=Relation(
                    operator=RelationOperator.NOT_EQUAL,
                    left=self._scalar(determinant),
                    right=self._integer(0),
                ),
                explanation=(
                    "A square matrix has an inverse exactly when its determinant is not zero."
                ),
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The determinant was computed exactly and is nonzero.",
                notes=(StepNote(label="Determinant", expression=self._scalar(determinant)),),
            ),
            self._step(
                rule="Form the adjugate over the determinant",
                before=query_expression(query),
                after=scaled,
                explanation=(
                    "The inverse equals the adjugate (transposed cofactor matrix) divided by the "
                    "determinant."
                ),
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="The adjugate formula for the inverse was applied.",
                notes=(StepNote(label="Adjugate", expression=self._matrix(adjugate)),),
            ),
            self._step(
                rule="Divide every entry by the determinant",
                before=scaled,
                after=self._matrix(backend_value),
                explanation="Divide each adjugate entry by the determinant and simplify.",
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The quotient matrix equals the backend inverse entry by entry.",
            ),
            self._step(
                rule="Check the inverse by multiplication",
                before=self._binary(
                    BinaryOperator.MULTIPLY,
                    self._matrix(matrix),
                    self._matrix(backend_value),
                ),
                after=self._matrix(identity),
                explanation="Multiplying the matrix by its inverse must give the identity matrix.",
                method=VerificationMethod.SUBSTITUTION,
                detail="The exact product was computed and equals the identity matrix.",
                notes=(StepNote(label="Identity matrix", expression=self._matrix(identity)),),
            ),
        )

    # Row reduction -------------------------------------------------------------------

    def _reduce_rows(self, matrix: sp.Matrix) -> _RowReduction | None:
        """Run exact Gauss-Jordan elimination and describe each elementary row operation."""
        if matrix.free_symbols:
            return None
        work = matrix.copy()
        steps: list[SolutionStep] = []
        pivots: list[int] = []
        pivot_row = 0
        for column in range(work.cols):
            if pivot_row >= work.rows:
                break
            source = next(
                (
                    row
                    for row in range(pivot_row, work.rows)
                    if not self._is_zero(work[row, column])
                ),
                None,
            )
            if source is None:
                continue
            if source != pivot_row:
                before = work.copy()
                work.row_swap(pivot_row, source)
                steps.append(self._swap_step(before, work, pivot_row, source))
            pivot_value = work[pivot_row, column]
            if pivot_value != sp.Integer(1):
                before = work.copy()
                work[pivot_row, :] /= pivot_value
                steps.append(self._scale_step(before, work, pivot_row, pivot_value))
            for row in range(work.rows):
                factor = work[row, column]
                if row == pivot_row or self._is_zero(factor):
                    continue
                before = work.copy()
                work[row, :] -= factor * work[pivot_row, :]
                steps.append(self._eliminate_step(before, work, row, pivot_row, factor))
            pivots.append(column)
            pivot_row += 1
        return _RowReduction(tuple(steps), work, tuple(pivots))

    def _swap_step(
        self,
        before: sp.Matrix,
        after: sp.Matrix,
        first: int,
        second: int,
    ) -> SolutionStep:
        return self._step(
            rule="Swap two rows",
            before=self._matrix(before),
            after=self._matrix(after),
            explanation=(
                f"Swap row {first + 1} with row {second + 1} to place a nonzero pivot in the "
                "current column."
            ),
            method=VerificationMethod.EXACT_ARITHMETIC,
            detail="Swapping two rows keeps the rows of the matrix unchanged as a set.",
            notes=(
                StepNote(
                    label="Rows swapped",
                    expression=SequenceExpression(
                        items=(self._integer(first + 1), self._integer(second + 1))
                    ),
                ),
            ),
        )

    def _scale_step(
        self,
        before: sp.Matrix,
        after: sp.Matrix,
        row: int,
        pivot_value: sp.Basic,
    ) -> SolutionStep:
        return self._step(
            rule="Scale a row to make its pivot 1",
            before=self._matrix(before),
            after=self._matrix(after),
            explanation=f"Divide row {row + 1} by its pivot so the pivot becomes 1.",
            method=VerificationMethod.EXACT_ARITHMETIC,
            detail="Dividing a row by a nonzero number is an invertible row operation.",
            notes=(StepNote(label="Divisor", expression=self._scalar(pivot_value)),),
        )

    def _eliminate_step(
        self,
        before: sp.Matrix,
        after: sp.Matrix,
        row: int,
        pivot_row: int,
        factor: sp.Basic,
    ) -> SolutionStep:
        return self._step(
            rule="Eliminate an entry in the pivot column",
            before=self._matrix(before),
            after=self._matrix(after),
            explanation=(
                f"Subtract {factor} times row {pivot_row + 1} from row {row + 1} to make the "
                "entry in the pivot column zero."
            ),
            method=VerificationMethod.EXACT_ARITHMETIC,
            detail="Adding a multiple of one row to another is an invertible row operation.",
            notes=(StepNote(label="Multiple subtracted", expression=self._scalar(factor)),),
        )

    def _verified_reduction(self, matrix: sp.Matrix) -> _RowReduction | None:
        reduction = self._reduce_rows(matrix)
        if reduction is None:
            return None
        expected, expected_pivots = matrix.rref()
        if reduction.pivots != tuple(expected_pivots) or not self._same_matrix(
            reduction.reduced, expected
        ):
            return None
        return reduction

    def _pivot_note(self, pivots: tuple[int, ...]) -> StepNote:
        return StepNote(
            label="Pivot columns",
            expression=SequenceExpression(
                items=tuple(self._integer(index + 1) for index in pivots)
            ),
        )

    def _rref_steps(
        self,
        _query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        reduction = self._verified_reduction(matrix)
        if reduction is None or not isinstance(backend_value, tuple):
            return ()
        reduced_expression = self._matrix(reduction.reduced)
        return (
            *reduction.steps,
            self._step(
                rule="Read the reduced row echelon form",
                before=self._matrix(matrix) if not reduction.steps else reduced_expression,
                after=reduced_expression,
                explanation=(
                    "Every pivot is 1 and is the only nonzero entry in its column, so the matrix "
                    "is in reduced row echelon form."
                ),
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The elimination result equals the backend reduced row echelon form.",
                notes=(self._pivot_note(reduction.pivots),),
            ),
        )

    def _rank_steps(
        self,
        _query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        reduction = self._verified_reduction(matrix)
        if reduction is None or backend_value != len(reduction.pivots):
            return ()
        reduced_expression = self._matrix(reduction.reduced)
        return (
            *reduction.steps,
            self._step(
                rule="Count the pivot rows",
                before=reduced_expression,
                after=self._integer(len(reduction.pivots)),
                explanation="The rank is the number of pivots in the reduced row echelon form.",
                method=VerificationMethod.EXACT_ARITHMETIC,
                detail="The pivot count equals the backend rank.",
                notes=(self._pivot_note(reduction.pivots),),
            ),
        )

    # Eigenvalues ---------------------------------------------------------------------

    def _eigenvalue_steps(
        self,
        query: Query,
        matrix: sp.Matrix,
        backend_value: object,
    ) -> tuple[SolutionStep, ...]:
        if not matrix.is_square or not is_object_mapping(backend_value):
            return ()
        eigenvalues: dict[sp.Basic, int] = {}
        for key, value in backend_value.items():
            if not isinstance(key, sp.Basic) or not isinstance(value, int):
                return ()
            eigenvalues[key] = value
        symbol = _EIGENVALUE_SYMBOL
        characteristic = sp.expand((matrix - symbol * sp.eye(matrix.rows)).det())
        if sum(eigenvalues.values()) != matrix.rows or not all(
            self._is_zero(characteristic.subs(symbol, root)) for root in eigenvalues
        ):
            return ()
        factored = sp.factor(characteristic)
        equation = Relation(
            operator=RelationOperator.EQUAL,
            left=self._scalar(characteristic),
            right=self._integer(0),
        )
        factored_equation = Relation(
            operator=RelationOperator.EQUAL,
            left=self._scalar(factored),
            right=self._integer(0),
        )
        ordered_roots = sorted(eigenvalues, key=str)
        return (
            self._step(
                rule="Form the characteristic equation",
                before=query_expression(query),
                after=equation,
                explanation=(
                    "Eigenvalues are the values of lambda for which det(M - lambda*I) equals zero."
                ),
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="The determinant of M - lambda*I was expanded exactly.",
                notes=(
                    StepNote(
                        label="Matrix M - lambda*I",
                        expression=self._matrix(matrix - symbol * sp.eye(matrix.rows)),
                    ),
                ),
            ),
            self._step(
                rule="Factor the characteristic polynomial",
                before=equation,
                after=factored_equation,
                explanation="Factor the polynomial so each root can be read from its factors.",
                method=VerificationMethod.SYMBOLIC_EQUIVALENCE,
                detail="Expanding the factored form recovers the characteristic polynomial.",
            ),
            self._step(
                rule="Read the eigenvalues and their multiplicities",
                before=factored_equation,
                after=SequenceExpression(items=tuple(self._scalar(root) for root in ordered_roots)),
                explanation=(
                    "Each root of the polynomial is an eigenvalue; its multiplicity is the number "
                    "of times its factor repeats."
                ),
                method=VerificationMethod.SUBSTITUTION,
                detail=(
                    "Each eigenvalue makes the characteristic polynomial zero and the "
                    "multiplicities add up to the size of the matrix."
                ),
                notes=tuple(
                    StepNote(
                        label=f"Multiplicity of {root}",
                        expression=self._integer(eigenvalues[root]),
                    )
                    for root in ordered_roots
                ),
            ),
        )
