# StepSolver project guidance

## Product goal

StepSolver must teach a person how to solve a problem, not merely expose a computer-algebra
result. An exact answer without a substantive, understandable derivation is a product defect.

## Derivation preferences

- Prefer reusable human methods—algebraic operations, substitution, factoring, differentiation
  rules, integration techniques, L’Hôpital’s rule, convergence tests, and generating methods—over
  jumping directly to a memorized identity.
- Identities may be shown as supporting notes or confirmation, but they should not replace the
  intermediate reasoning when a standard classroom method is available.
- Show one meaningful transformation per step. Do not collapse several operations into an opaque
  “simplify,” “collect,” or “compute exact result” step.
- Explicitly expose why a method applies: indeterminate forms, convergence conditions, domain
  restrictions, nonzero divisors, and other prerequisites belong in the worked solution.
- Display the operation being performed. For example, show terms added to both sides, quotients
  used to divide both sides, derivatives of numerator and denominator, substitutions, and actual
  cancellations before presenting the simplified result.
- Keep the final symbolic identity as a check when it helps, but optimize the main sequence for
  intuition, readability, and correspondence with handwritten mathematics.
- Use structured inline-math fragments for formulas inside prose explanations so the frontend
  renders them as LaTeX. Do not encode formulas with ad-hoc Unicode superscripts or ASCII syntax.
- Use `match` statements for genuine closed variants or operation families when they make the
  dispatch clearer; do not force pattern matching where ordinary conditions are simpler.

## Regression expectations

- Every supported exact result must have nonempty human steps and must avoid the opaque
  `Compute exact result` fallback.
- Add regressions for the reported expression and nearby structural or parameter variations.
- Tests should verify the named method, intermediate transformations, mathematical notes,
  applicability conditions, and absence of backend-specific or plaintext-math leakage.
- Fuzz or systematically vary coefficients, shifts, bounds, functions, and equivalent forms when
  extending a derivation family.

## Frontend preferences

- Mathematical content must use the math renderer, including formulas embedded in explanations.
- Prefer accessible SVG icons over Unicode glyphs used as interface icons.
- Loading progress should only appear when relevant, describe real work, and disappear when the
  work completes.
- Preserve responsive layouts, keyboard behavior, readable spacing, and accessible labels.

## Quality gate

Before committing, run the relevant focused regressions, the complete Python suite with the
coverage threshold, frontend tests and production build, Ruff, mypy, Pyright, ty, Poetry package
validation, and `git diff --check`.
