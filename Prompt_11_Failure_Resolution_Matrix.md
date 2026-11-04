# Prompt 11 — Failure Resolution Matrix

Maps every Prompt 11 requirement to its baseline evidence, root cause, fix, final
evidence, and regression tests.

**Baseline:** `7d17871299c9e1a8393c503033c7b3a063616a25`
**Branch:** `prompt11-stabilisation`
**Implementation commit:** `459181c47b5ea5a3e07f4961ae7e6af62aa3dd0c`
**Documentation finalisation:** docs-only follow-up commit (branch HEAD at delivery is reported
in the Prompt 11 delivery message)

Legend: ✅ fixed/verified · ➖ already correct, regression-locked · ⛔ blocked (fixture absent)

---

## 1. Required correctness cases

| Case | Required contract | Baseline | Root cause | Fix | Final evidence | Regression test | Status |
|---|---|---|---|---|---|---|---|
| **C15** | Empty **real** solution set under the benchmark real-domain contract; complex solving still available when explicitly requested | Real domain already returned the empty set; default returned `{-2i, 2i}` | None (contract already honoured); needed explicit locking | No behaviour change. Verified the real/complex split and documented the README default | `domain="real"` → `no solution (empty set)`, with discriminant steps; no `domain` → `{-2i, 2i}` | `tests/test_domain_algebra_constraints.py::test_real_domain_discards_nonreal_quadratic_roots`, `::test_complex_domain_keeps_both_imaginary_roots`, `::test_omitted_domain_preserves_complex_polynomial_default` | ➖ |
| **C10** | Canonicalise identity to **all reals** | all real | None | No behaviour change; regression-locked | `all real`; derivation 10/10 | `tests/test_identity_nsolve_dsolve_syntax.py::test_identity_equation_serializes_as_a_universal_solution`, `::test_contradiction_is_not_reported_as_a_universal_solution` | ➖ |
| **C14** | Canonicalise expanded identity to **all reals** | all real | None | No behaviour change; regression-locked | `all real`; derivation 10/10 | same identity suite (universal-solution + contradiction control) | ➖ |
| **L08** | **All reals excluding `x != 1`**, restriction preserved structurally | all real; restriction present | None in the result | No behaviour change; explicit structural-restriction assertions added | `actual=all real`; `restrictions_reported` contains the denominator note, `x - 1 != 0`, and `x != 1` | `tests/test_identity_nsolve_dsolve_syntax.py::test_identity_with_a_denominator_keeps_its_exclusion` | ➖ |

**Note on L08 `domain_check.satisfied`:** the benchmark records `null` because the case
declares no external required-restriction list. The restriction is not lost: it is carried
in `restrictions_reported` and in the derivation's introduced constraints.

---

## 2. Native numeric cases (nsolve)

| Case | Required contract | Baseline | Root cause | Fix | Final evidence | Regression test | Status |
|---|---|---|---|---|---|---|---|
| **K03** | Native `nsolve`, preserve guess/variable/precision, serialisable `NumericResult`, honest residual/convergence, never `UnexpectedResultType` | `0.739085133215161` | None | No engine change; locked with residual/iterate assertions | `[0.739085133215161]`; Newton trace; residual < 1e-12 | `tests/test_numeric_solver.py::test_nsolve_k03_returns_verified_newton_trace` | ➖ |
| **K08** | same | `2.09455148154233` | None | No engine change | `[2.09455148154233]` | `tests/test_identity_nsolve_dsolve_syntax.py::test_nsolve_native_queries_return_verified_roots` (parametrised) | ➖ |
| **K09** | same | `0.567143290409781` | None | No engine change | `[0.567143290409781]` | same parametrised test | ➖ |
| **K10** | same | `0.523598775598299` | None | No engine change | `[0.523598775598299]` | same parametrised test | ➖ |
| **K11** | same | `2.71828182845905` | None | No engine change | `[2.71828182845905]` | same parametrised test + `tests/test_ln_alias.py` | ➖ |

Guards also cover: imaginary guess rejected, zero-derivative start typed failure,
nonconvergence bounded and typed, identity residual typed failure, missing variable typed
failure (`tests/test_numeric_solver.py`).

---

## 3. Native ODE cases (dsolve)

| Case | Required contract | Baseline | Root cause | Fix | Final evidence | Regression test | Status |
|---|---|---|---|---|---|---|---|
| **J05** | Parse/dispatch `diff(y(x), x)` natively; solution family with arbitrary constant; substitution check | `Eq(y(x), C1*exp(-2*x))`, overrides = false | None | No engine change; locked | `Eq(y(x), C1*exp(-2*x))`, overrides = false, derivation 10/10 | `tests/test_identity_nsolve_dsolve_syntax.py::test_dsolve_accepts_sympy_derivative_notation`, `::test_dsolve_isolated_serialization_reports_an_equation` | ✅/➖ |
| **J06** | Parse/dispatch second-order `diff(y(x), x, 2)` natively | `Eq(y(x), (C1 + C2*exp(x))*exp(x))`, overrides = false | None | No engine change; locked | same answer, overrides = false, derivation 10/10 | same dsolve suite + `::test_native_tuple_derivative_order_matches_the_three_argument_form` | ✅/➖ |
| **J01** | Eliminate the legacy override; dispatch natively | correct answer but **adapter override = true** | Harness honoured legacy `adapter_inputs` for dsolve instead of building the query from case metadata | Harness-only fix: all dsolve queries built from `{ode, func, var}`; legacy dsolve overrides ignored. Engine already parsed the form | `Eq(y(x), C1*exp(3*x))`, **overrides = false** | dsolve parsing suite; adapter verified via V2 rerun | ✅ |
| **J02** | Eliminate the legacy override; dispatch natively (2nd order) | correct answer but **adapter override = true** | same | same | `Eq(y(x), C1*sin(x) + C2*cos(x))`, **overrides = false** | same | ✅ |
| Hard ODE overrides | Applicable Hard ODE cases should not rely on overrides | ⛔ fixture absent | — | — | cannot verify | — | ⛔ |

**Verified syntax forms** (probed directly against the engine):

- `diff(y(x), x)` ✅
- `diff(y(x), x, 2)` ✅
- `Derivative(y(x), x)` ✅
- `Derivative(y(x), (x, 2))` ✅
- `diff(y(x), x, x)` ❌ unsupported (documented limitation; use one of the above)

**Adapter outcome:** V2 overrides 2/141 (1.42 %) → **0/141 (0.00 %)**.

---

## 4. Hard & Adversarial cases

| Case / area | Required contract | Baseline | Fix | Final evidence | Status |
|---|---|---|---|---|---|
| **R06** | `integrate(1/(x^2-1), x)` uses real `log(abs(...))`, reports `x != -1` and `x != 1`; G04/G08 stay correct | `log(x - 1)/2 - log(x + 1)/2 + C` — invalid for real `x < -1` | Generic real-log partial-fraction matcher + interval-aware verification + `Abs` derivation terms and structural constraints | `-log(abs(x + 1))/2 + log(abs(x - 1))/2 + C`; `x != -1`, `x != 1`; variants `1/(x^2-4)`, `1/(x^2+x)`, `1/(x(x+1))`, `3/(x^2-9)`, `(x+2)/(x^2-1)` all correct | ✅ |
| **T06** | Structured singular-matrix result with explanatory steps; do not conflate undefined/divergent/no-solution/singular | `Unsolved: Matrix det == 0; not invertible.` | `SingularMatrixResult`, zero-det precheck, determinant/rank steps, end-to-end `"singular"` status | `SingularMatrixResult`; det = 0; rank 1 < 2; "inverse does not exist"; payload/isolated/browser status `"singular"` | ✅ (implementation; official T06 row unavailable) |
| **U01–U07** | Applicable Hard U-cases must pass | ⛔ no fixture | — | — | ⛔ |
| **Hard N-/O-suite overrides** | Hard ODE/integral cases should be native | ⛔ no fixture | — | — | ⛔ |
| **Hard 89 overall** | Run and report separately | ⛔ fixture absent | — | Documented blocker; no score claimed | ⛔ |

Search evidence for the absent fixture is summarised in
`docs/benchmarks/prompt11/hard_89_BLOCKED.md`.

---

## 5. Regression guard cases (must remain passing)

| Case | Required | Baseline | Final | Test | Status |
|---|---|---|---|---|---|
| **G04** | `integrate(1/x, x)` = real log with pole exclusion | `log(abs(x)) + C`, `x != 0` | unchanged | `tests/test_pedagogy.py::test_reciprocal_integral_uses_absolute_value_and_preserves_the_pole`, `tests/test_real_log_antiderivatives.py::test_g04_and_g08_keep_their_absolute_logarithms` | ✅ |
| **G08** | `integrate(2/(2x+1), x)` = real log with shift/pole exclusion | `log(abs(2*x + 1)) + C`, `x != -1/2` | unchanged; `domain_check.satisfied = true` | same tests + `::test_reciprocal_integral_handles_shifted_linear_denominators` | ✅ |
| Nonsingular inverse | Exact matrix must be unaffected by the singular path | `[[-2, 1], [3/2, -1/2]]` | unchanged | `tests/test_singular_matrix.py::test_nonsingular_inverse_still_returns_the_exact_matrix` | ✅ |
| C15 complex default | Omitting `domain` keeps complex roots | `{-2i, 2i}` | unchanged | `tests/test_domain_algebra_constraints.py::test_omitted_domain_preserves_complex_polynomial_default` | ✅ |

---

## 6. Serialization and contract surface

| Surface | Requirement | Change | Test |
|---|---|---|---|
| `SolveResult` union | New typed singular outcome | `SingularMatrixResult` added and exported | `tests/test_singular_matrix.py` |
| `solve_payload(...)` | Distinct status, honest reason, latex | `status="singular"`, `result_latex="\text{No inverse: singular matrix}"`, `result=None` | `::test_singular_inverse_payload_has_a_distinct_status` |
| `solve_isolated(...)` | Agent-facing status preserved | literal extended with `"singular"` | `::test_singular_inverse_isolated_serialization_keeps_the_status` |
| Web `SolveResponse` | Browser contract accepts singular | status literal extended | `tests/test_web.py` |
| Frontend | Render singular distinctly from unsolved | "Singular matrix" status label; build regenerated | `frontend/*.test.mjs`, `npm run build` |
| ASCII formatter | Human-readable singular explanation | `Singular matrix: ...` line | `::test_singular_inverse_returns_a_typed_matrix_outcome` |

---

## 7. Aggregate verification

| Check | Result |
|---|---|
| Core 72 | 72/72 strict, 72/72 covered, 72/72 derivations, mean final derivation 9.2639, 21/21 domain-sensitive, warm median 33.461 ms |
| V2 Stress 141 | 141/141 strict, 141/141 covered, 0 overrides, 0 timeouts, 0 crashes, overall 95.3241 |
| Hard 89 | ⛔ blocked — fixture absent, no score claimed |
| Full Python suite | 831 passed, coverage 90.27 % |
| Ruff check / format | PASS |
| mypy | PASS (103 files) |
| Pyright | 0 errors, 56 missing-module-source warnings |
| ty | PASS |
| Poetry metadata | PASS |
| Frontend tests / build | 26/26 PASS / PASS |
| `git diff --check` | PASS |
| Gates G1–G4 | PASS / PASS / PASS / PASS |
| Production recommendation | YES (within available evidence) |

---

## 8. Final commit

- Branch: `prompt11-stabilisation`
- Baseline: `7d17871299c9e1a8393c503033c7b3a063616a25`
- Implementation commit: `459181c47b5ea5a3e07f4961ae7e6af62aa3dd0c`
- Documentation finalisation: docs-only follow-up commit; branch HEAD at delivery is
  reported in the Prompt 11 delivery message.
- Not pushed.

---

## 9. Open items

1. **Hard 89 fixture** — must be supplied to close R06/T06/U01–U07 against official rows.
2. **Core 72 standalone runner** — currently measured from the embedded V2 subset.
3. **T06 identity** — implementation is generic; exact case identity cannot be confirmed without the Hard fixture.
4. **Peak RSS** — harness reports `0.0 MB`; no real memory measurement available.
5. **`diff(y(x), x, x)`** — unsupported repeated-variable form; documented, with two supported alternatives.
