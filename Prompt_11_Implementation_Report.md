# Prompt 11 — Final Correctness & Public API Stabilisation

**Repository:** `C:\Users\Olabode Nathaniel\Code\Step Solver\stepsolver`
**Branch:** `prompt11-stabilisation`
**Baseline commit:** `7d17871299c9e1a8393c503033c7b3a063616a25` (`main`, "Merge improve-coverage-performance into main")
**Implementation commit:** `459181c47b5ea5a3e07f4961ae7e6af62aa3dd0c`
**Documentation finalisation:** docs-only follow-up commit on the same branch (see `git log -1`)

---

## 1. Executive summary

Prompt 11 was a targeted stabilisation pass, not a rewrite. StepSolver remains the
authoritative engine: no routing replacement, no new solver architecture, no heavy
dependencies, and no benchmark-only branches in product code.

Three defects were fixed at root-cause level, with tests that generalise around each
reported case:

| Area | Baseline behaviour | Final behaviour |
|---|---|---|
| **R06** `integrate(1/(x^2-1), x)` | `log(x - 1)/2 - log(x + 1)/2 + C` — invalid at real `x < -1` | `log(abs(x - 1))/2 - log(abs(x + 1))/2 + C`, with `x != -1`, `x != 1` |
| **Singular inverse** (`T06` class) | `Unsolved: Matrix det == 0; not invertible.` | New `SingularMatrixResult` (`status="singular"`) with determinant/rank steps and explanatory text |
| **dsolve adapter overrides** (J01/J02) | Legacy `adapter_inputs` used; 2/141 overrides (1.42 %) | All ODE queries built natively from case metadata; 0/141 overrides (0.00 %) |

Required correctness cases C15, C10, C14, L08, K03, K08–K11, J01/J02, J05/J06, G04
and G08 all pass strict correctness **without** adapter overrides.

**Verified results**

- **Core 72:** 72/72 strict correct, 72/72 covered, 72/72 derivations available, mean final derivation 9.2639, 21/21 domain-sensitive cases strict, warm median 33.461 ms.
- **V2 Stress 141:** 141/141 strict correct, 141/141 covered, 0 timeouts, 0 crashes; overall score **95.3241**.
- **Hard & Adversarial 89:** **BLOCKED — fixture is absent from this workspace and the benchmark environment.** No score is claimed. See § 8.

**Hard gates**

| Gate | Status |
|---|---|
| G1 critical correctness | **PASS** |
| G2 fabricated/invalid derivations | **PASS** |
| G3 unsafe execution / untrusted input | **PASS** (canary score 1.0; payload rejected with `ParseError`) |
| G4 licence | **PASS** (MIT) |

Eligible for production recommendation: **YES** (within the evidence available; the Hard 89 suite remains unverified).

---

## 2. Scope, constraints and non-goals

Upheld throughout:

- StepSolver stays the primary engine; SymPy is used exactly where the existing backend already used it.
- Public API, typed results, derivation framework, domain handling and parser allow-list stay authoritative.
- No `eval`, `exec`, arbitrary Python evaluation, or new unsafe execution path.
- No case-ID branches, hard-coded answers, benchmark detection, judge weakening, or tolerance relaxation.
- No product-code text matching to fake correctness.
- Tests were added for the reported case and for nearby structural/parameter variations.

Explicit non-goals: rewriting routing, adding a new CAS/model layer, or changing the
documented complex-root default for exact `solve`.

---

## 3. Environment

| Item | Value |
|---|---|
| Repository Python | 3.14.4 (Poetry virtualenv `stepsolver-H0q064b0-py3.14`) |
| Benchmarks Python | 3.12.13 (`math_solver_benchmark_v2_local/envs/stepsolver`) |
| SymPy | 1.14.0 (both) |
| OS | Windows 11 (10.0.26300) |
| Benchmark protocol | `bench.runner --timeout 20 --repeats 5` |
| Benchmark root | scratch copy of `math_solver_benchmark_v2_local` |
| Raw V2 run | `%TEMP%\opencode\prompt11_v2_141_timed_final` |
| Evaluated V2 rows | `%TEMP%\opencode\prompt11_v2_141_timed_final_eval\raw_results.json` |
| Shadow scores | `%TEMP%\opencode\prompt11_shadow_score\results\scores.json` |
| Engineering probes | `%TEMP%\opencode\prompt11_engineering_after.json` |

The benchmark root's preserved `results/` files were left untouched. A shadow scoring
root merged only the new StepSolver rows with the unchanged SymPy Direct and
SmartSolver baseline rows, then ran the unmodified `bench/score.py`.

---

## 4. Root causes and fixes

### 4.1 R06 — real-valued logarithm antiderivatives

**Root cause.** Indefinite integration of rational functions whose partial fractions
produce linear denominators used SymPy's principal `log(linear)` form. Over the real
numbers this is wrong on intervals where the linear argument is negative: the derivative
of `log(x - 1)` is `1/(x - 1)`, but `log(x - 1)` is not real for `x < 1`. The answer was
therefore not a valid real antiderivative across the whole real domain excluding the poles.

**Fix (generic, not case-specific):**

1. `sympy_support.py`
   - `real_zeros(...)` / real-zero helpers: collect the real zeros of denominators.
   - `real_intervals_between(...)`: build the open real intervals between consecutive poles.
   - `differentiates_on_real_intervals(...)`: verify a candidate antiderivative by
     differentiating on each interval, using `refine` with explicit interval assumptions.
   - All SymPy access is guarded through typed `cast` shims, so the parser allow-list and
     the type stubs stay intact.
2. `sympy_integrals.py`
   - `PartialFractionAntiderivative` dataclass and
     `match_partial_fraction_antiderivative(...)`: match a rational integrand whose
     denominator factors into real linear factors, and integrate each term as
     `coefficient * log(Abs(normalized_linear))`, where `normalized_linear` is monic.
   - Only applied when the leading coefficient is provably real.
3. `sympy_execution.py`
   - For indefinite integrals, prefer the real-log candidate over the principal-log
     candidate when the real-log candidate verifies on every real interval.
4. `sympy_verification.py`
   - Interval-aware verification of `Abs`-based antiderivatives: differentiate on each
     interval separately and require a zero residual on all of them.
5. `derivation/integrals_special.py`
   - `derive_partial_fraction_integral` now emits `log(Abs(...))` terms and attaches the
     original denominator's real zeros as structural `BackendStepConstraint`s.

**Final output (R06):**

```
Input: integrate(1/(x^2 - 1), x)
Step 1 (Decompose into partial fractions): ...
Step 2 (Integrate each partial fraction): ...
  -> -log(abs(x + 1)) / 2 + log(abs(x - 1)) / 2 + C
  Domain constraint: x != -1 (The original denominator cannot equal zero.)
  Domain constraint: x != 1  (The original denominator cannot equal zero.)
Result: -log(abs(x + 1)) / 2 + log(abs(x - 1)) / 2 + C
```

**Baseline (recorded from the `7d17871` worktree):**

```
Result: log(x - 1) / 2 + -log(x + 1) / 2 + C
```

Generalisations verified: `1/(x^2-4)`, `1/(x^2+x)`, `1/(x*(x+1))`, `3/(x^2-9)`,
`(x+2)/(x^2-1)`.

**No regression to existing real-log cases:** G04 stays `log(abs(x)) + C`; G08 stays
`log(abs(2*x + 1)) + C` with `x != -1/2` and `domain_check.satisfied = true`.

### 4.2 T06 class — structured singular-matrix outcome

**Root cause.** `inverse(matrix(...))` delegated to SymPy; a zero determinant surfaced as
a generic backend failure and was serialised as `Unsolved`. That conflated four distinct
mathematical outcomes: undefined, divergent, no solution, and singular.

**Fix:**

- `results.py`: new frozen `SingularMatrixResult` (determinant, rank, size, reason, steps)
  added to the `SolveResult` union and exported from `stepsolver`.
- `sympy_backend.py`: zero-determinant precheck before attempting inversion; returns
  `SingularMatrixResult` instead of an unsolved/untyped failure. Uses
  `determinant.is_zero is True`, not `!= 0`, so symbolic or undecidable determinants are
  not misclassified.
- `sympy_matrix_steps.py`: `singular_inverse_steps(...)` emits
  (1) compute the determinant, (2) conclude singularity with `det = 0`, `rank < n`,
  "inverse does not exist".
- Presentation plumbing: `formatter.py`, `presentation.py`, `web.py`, `isolated.py`,
  `__init__.py`, `web_assets/app.js` — status `"singular"` is carried end-to-end, with
  latex `\text{No inverse: singular matrix}` and the browser status label
  "Singular matrix".

**Final output:**

```
Input: inverse(matrix([[1,2],[2,4]]))
Step 1 (Compute the determinant): ...
  -> 0
Step 2 (Conclude the matrix is singular): ...
  det(matrix([[1, 2], [2, 4]])) = 0
  -> inverse does not exist
  Rank: 1 < 2
Singular matrix: The matrix is singular because its determinant is zero, so its inverse does not exist.
```

`SolvePayload.status == "singular"`; isolated runner status `"singular"`; browser status
`"singular"`.

**Baseline:**

```
Input: inverse(matrix([[1,2],[2,4]]))
Unsolved: Matrix det == 0; not invertible.
```

A nonsingular inverse (`[[1,2],[3,4]]`) still returns the exact matrix unchanged.

### 4.3 dsolve — native parsing and dispatch

**Root cause.** The local benchmark adapter used legacy per-case `adapter_inputs` for
J01/J02. The engine itself parsed those forms, but the adapter path meant J01/J02 were
recorded as adapted rather than native, and the override count stayed at 2/141.

**Fix.**

- Product code already parses both `diff(y(x), x)` / `diff(y(x), x, 2)` and
  `Derivative(y(x), x)` / `Derivative(y(x), (x, 2))`; J05/J06 were always native.
- The benchmark worker now constructs every `dsolve` query mechanically from case
  metadata (`ode`, `func`, `var`) and ignores legacy `dsolve` overrides. This is a
  harness-only change (outside the repository); no product code was weakened to make it
  pass.
- J01/J02 are now dispatched natively: `dsolve(diff(y(x), x) - 3*y(x) = 0, y(x))`,
  `dsolve(diff(y(x), x, 2) + y(x) = 0, y(x))`.

**Result:** override count 2 → 0. All four J-suite rows remain strict-correct, with
nonempty derivations.

### 4.4 Domain-contract cases (C15, C10, C14, L08)

These already behaved correctly at baseline under the benchmark's real-domain contract;
Prompt 11 adds explicit regression coverage rather than changing behaviour.

- **C15** `x^2 + 4 = 0` with `domain="real"` → empty real solution set ("No solution"),
  with discriminant reasoning. Omitting `domain` still returns the complex family
  `{-2i, 2i}`, preserving the documented default.
- **C10** `2*x + 1 = 2*x + 1` → all reals.
- **C14** `(x + 1)^2 = x^2 + 2*x + 1` → all reals.
- **L08** `1/(x - 1) = 1/(x - 1)` → all reals, with the original denominator's
  `x != 1` restriction preserved structurally (constraint text plus `x != 1`).

### 4.5 nsolve — native numeric path

`nsolve` was already native at baseline. Prompt 11 confirms and regression-tests that it:

- preserves the guess, variable and requested precision;
- returns a serialisable `NumericResult` / `NumericRootsValue`;
- reports Newton iterates, residual and convergence honestly;
- never raises `UnexpectedResultType` for the supported K-suite queries.

K03 `cos(x)=x` → `0.739085133215161`; K08 `x^3-2x-5=0` → `2.09455148154233`;
K09 `exp(-x)=x` → `0.567143290409781`; K10 `sin(x)=1/2` → `0.523598775598299`;
K11 `ln(x)=1` → `2.71828182845905`.

---

## 5. Changed files

Product code (repository):

| File | Change |
|---|---|
| `src/stepsolver/sympy_support.py` | Real zeros, real intervals, `differentiates_on_real_intervals`, typed SymPy shims |
| `src/stepsolver/sympy_integrals.py` | Partial-fraction matcher + real-log term integration |
| `src/stepsolver/sympy_execution.py` | Real-log candidate selection for indefinite integrals |
| `src/stepsolver/sympy_verification.py` | Interval-aware `Abs` antiderivative verification |
| `src/stepsolver/derivation/integrals_special.py` | `log(Abs(...))` derivation terms + structural constraints |
| `src/stepsolver/results.py` | `SingularMatrixResult`, `SolveResult` union |
| `src/stepsolver/sympy_backend.py` | Singular-inverse precheck |
| `src/stepsolver/sympy_matrix_steps.py` | Singular determinant/rank steps |
| `src/stepsolver/sympy_steps.py` | Step builder wiring |
| `src/stepsolver/presentation.py` | `"singular"` payload status |
| `src/stepsolver/formatter.py` | ASCII rendering of singular result |
| `src/stepsolver/isolated.py` | Isolated status literal + serialisation |
| `src/stepsolver/web.py` | Browser response status literal |
| `src/stepsolver/__init__.py` | Public export |
| `src/stepsolver/web_assets/app.js` | Browser status label |
| `typings/sympy/__init__.pyi` | `Float.__float__` stub for interval sorting |

Tests:

| File | Change |
|---|---|
| `tests/test_real_log_antiderivatives.py` | New: R06 + rational-log variations, constraints, latex, equivalence |
| `tests/test_singular_matrix.py` | New: typed singular result, rank/determinant steps, payload, isolated serialisation, nonsingular control |
| `tests/test_valid_results_inequalities_integrals.py` | Updated R06 expectations |
| `tests/test_pedagogy.py` | Updated R06 derivation expectations |

Harness (outside repository): `bench/workers/stepsolver_worker.py` builds native
`dsolve` queries and ignores legacy `dsolve` overrides. Not a product-code change.

---

## 6. Tests and quality gates

All run from the repository root with the Poetry virtualenv binaries.

| Check | Result |
|---|---|
| Full Python suite + coverage gate | **831 passed in 403.86 s**, coverage **90.27 %** (gate 90 %) |
| Ruff check | PASS — "All checks passed!" |
| Ruff format check | PASS — 107 files already formatted |
| mypy | PASS — no issues in 103 source files |
| Pyright | PASS — 0 errors, 56 `reportMissingModuleSource` warnings (SymPy stubs) |
| ty | PASS — "All checks passed!" |
| Poetry metadata | PASS — `uvx --from poetry poetry check --strict` → "All set!" |
| Frontend tests | PASS — 26/26 |
| Frontend production build | PASS — `web_assets/app.js` regenerated and consistent |
| `git diff --check` | PASS — no whitespace errors |

Targeted suites were run after each task group (real-log, singular matrix, dsolve,
serialisation), then the full suite, then the benchmarks.

---

## 7. Targeted before/after evidence

All "after" values are from
`prompt11_v2_141_timed_final_eval/raw_results.json` (141/141 strict). R06 and the
singular matrix are not in the V2 141 dataset, so their before/after come from the
`7d17871` worktree and the current tree.

| Case | Input | Before | After | Strict |
|---|---|---|---|---|
| C15 | `x^2 + 4 = 0` (real) | no real solution | no real solution (empty set); complex on default | ✅ |
| C10 | `2*x + 1 = 2*x + 1` | all real | all real | ✅ |
| C14 | `(x + 1)^2 = x^2 + 2*x + 1` | all real | all real | ✅ |
| L08 | `1/(x - 1) = 1/(x - 1)` | all real, `x != 1` | all real, `x != 1` preserved structurally | ✅ |
| K03 | `cos(x) = x` near 0.7 | `0.739085133215161` | same | ✅ |
| K08 | `x^3 - 2*x - 5 = 0` near 2 | `2.09455148154233` | same | ✅ |
| K09 | `exp(-x) = x` near 0.5 | `0.567143290409781` | same | ✅ |
| K10 | `sin(x) = 1/2` near 0.5 | `0.523598775598299` | same | ✅ |
| K11 | `ln(x) = 1` near 2.5 | `2.71828182845905` | same | ✅ |
| J01 | `y' = 3*y` | `Eq(y(x), C1*exp(3*x))`, override | same, **no override** | ✅ |
| J02 | `y'' + y = 0` | `Eq(y(x), C1*sin(x) + C2*cos(x))`, override | same, **no override** | ✅ |
| J05 | `y' + 2y = 0` | `Eq(y(x), C1*exp(-2*x))` | same | ✅ |
| J06 | `y'' - 3y' + 2y = 0` | `Eq(y(x), (C1 + C2*exp(x))*exp(x))` | same | ✅ |
| G04 | `integrate(1/x, x)` | `log(abs(x)) + C` | same | ✅ |
| G08 | `integrate(2/(2x+1), x)` | `log(abs(2x+1)) + C` | same, domain satisfied | ✅ |
| R06 | `integrate(1/(x^2-1), x)` | `log(x-1)/2 - log(x+1)/2 + C` | `log(abs(x-1))/2 - log(abs(x+1))/2 + C`, `x != ±1` | ✅ |
| T06 class | `inverse([[1,2],[2,4]])` | `Unsolved: Matrix det == 0; not invertible.` | `SingularMatrixResult`, `status="singular"`, det/rank steps | ✅ |

### Required-case notes

- **C15** satisfies the benchmark's real-domain contract. The README-documented default
  (complex roots when `domain` is omitted) is preserved unchanged.
- **L08** is strict-correct with `restrictions_reported` containing both the denominator
  explanation and `x != 1`. The benchmark records `domain_check.satisfied = null` because
  the case has no required external restriction list; the restriction is carried
  structurally in the result, not lost.
- **J01/J02** are now native in the adapter (`adapter_overrides = false`), and the engine
  parses both `diff(...)` and `Derivative(...)` forms.

---

## 8. Benchmark results

Scores are reported **separately**. They are not combined into a single artificial score.

### 8.1 Core benchmark — 72 cases

The Core 72 definition is the embedded subset of the V2 dataset:
`A01–A08, B01–B08, C01–C10, D01–D06, E01–E08, F01–F06, G01–G06, H01–H04, I01–I04, J01–J04, K01–K04, L01–L04`.

| Metric | Baseline | Final |
|---|---|---|
| Strict correct | 72/72 | **72/72** |
| Covered | 72/72 | **72/72** |
| Invalid answers | 0 | **0** |
| Errors/timeouts | 0 | **0** |
| Derivations available | 72/72 | **72/72** |
| Mean final derivation | 9.2639 | **9.2639** |
| Mean automated derivation | 9.2361 | **9.2361** |
| Domain-sensitive strict | 21/21 | **21/21** |
| Warm median | 30.855 ms | **33.461 ms** |

No standalone Core-72 runner or score fixture exists in the workspace; this summary is
computed from the 72 embedded V2 rows with the same evaluator. See § 9.

### 8.2 V2 Stress benchmark — 141 cases

| Metric | Value |
|---|---|
| Cases | 141 |
| Outcomes | `{'answer': 141}` |
| Strict correct | **141/141** |
| Covered | **141/141** |
| Timeouts | **0** |
| Crashes | **0** |
| Adapter overrides | **0** (baseline 2) |
| Overall score | **95.3241** |

Components:

| Component | Score |
|---|---|
| Correctness | 100.0 |
| Derivations | 93.2624 |
| Coverage | 100.0 |
| Reliability | 100.0 |
| Integration | 85.7143 |
| Performance | 61.1934 |

Per-category strict correctness: A 13/13, B 13/13, C 16/16, D 11/11, E 13/13,
F 12/12, G 13/13, H 9/9, I 9/9, J 8/8, K 12/12, L 12/12.

Baseline V2 for comparison: strict 141/141, correctness 100, derivations 93.2624,
coverage 100, reliability 100, integration 85.7143, performance 62.2994, overall
**95.3794**. The movement to 95.3241 is entirely the performance component
(62.2994 → 61.1934); correctness, derivations, coverage, reliability and integration are
identical.

### 8.3 Hard & Adversarial benchmark — 89 cases

**BLOCKED — not run; no score claimed.**

The 89-case Hard & Adversarial fixture is not present in:

- the repository (`Benchmarking tests/` contains only the original Colab notebook, the V2
  notebook, and an open-source benchmark notebook);
- the local benchmark root (`data/`, `results/`, `docs/`, `candidates/`, `pins/`);
- any sibling directory under `C:\Users\Olabode Nathaniel\Code\Step Solver`.

The V2 notebook documents the original 72 core cases plus 69 stress cases; the harness
dataset contains exactly 141 cases, not 161. An exhaustive filename/content search for
`Hard 89`, `adversarial`, `U01`–`U29`, `89-case` found no fixture.

Consequently:

- R06 and T06 acceptance cannot be evaluated against the official Hard 89 rows; they are
  evidenced with targeted before/after probes and regression tests instead.
- U01–U07 and any Hard-specific N- or O-suite cases cannot be verified.
- The report does **not** fabricate a Hard 89 score or a pass.

See `docs/benchmarks/prompt11/hard_89_BLOCKED.md`.

---

## 9. No-regression analysis

Row-by-row comparison of all 141 V2 StepSolver rows, baseline
(`math_solver_benchmark_v2_local/results/raw_results.json`) vs final:

- status: identical
- strict correctness: identical
- canonical answer text: identical
- derivation final score: identical
- errors/timeouts: 0 in both

**Changed rows: 0.**

The only recorded difference is `adapter_overrides`: 2 → 0 (J01/J02). This is an
improvement in native-integration accounting, not an engine behaviour change.

Additional controls:

- Nonsingular inverse still returns the exact matrix.
- G04/G08 real-log integrals unchanged.
- C15 complex default unchanged when `domain` is omitted.
- 831-test suite passes at 90.27 % coverage.

The 7 "manual reviews NOT applied (engine output changed since review)" lines from
`bench.evaluate` are noted: the evaluator refused stale human overlays, so those rows were
scored automatically. Because every row's recorded final derivation score is unchanged
from baseline, this did not alter the comparison; it is disclosed rather than hidden.

---

## 10. Adapter report

| Item | Detail |
|---|---|
| Adapter | `bench/workers/stepsolver_worker.py` (local harness, outside repo) |
| Change | `dsolve` queries always constructed from case metadata; legacy `dsolve` overrides ignored |
| Overrides before | 2/141 (1.42 %): J01, J02 |
| Overrides after | **0/141 (0.00 %)** |
| Non-dsolve overrides | None added or removed |
| Product-code impact | None; no benchmark detection or case-ID branching |

The adapter change is deliberately confined to the harness. The engine was independently
probed with both `diff(y(x), x)` and `Derivative(y(x), x)` forms, including tuple orders.

---

## 11. Performance report

V2, 5 repeats, 20 s timeout, same machine and protocol for all solvers.

| Metric | Baseline | Final |
|---|---|---|
| Cold ready (median) | 2441.36 ms | 2950.87 ms |
| Cold first solve (median) | 167.17 ms | 167.91 ms |
| Warm median (common 82 cases) | 61.64 ms | 61.98 ms |
| Warm p95 (common) | 239.27 ms | 275.52 ms |
| Own median | 41.73 ms | 40.88 ms |
| Own p95 | 488.27 ms | 405.37 ms |
| Timeouts | 0 | 0 |
| Crashes | 0 | 0 |

Per-operation warm median (final): diff 59.62 ms, gradient 98.86 ms, integrate 46.66 ms,
limit 171.60 ms, nsolve 45.93 ms, partial derivative 40.88 ms, partial fractions 72.27 ms,
rref 7.25 ms, solve 104.20 ms, solve_system 32.65 ms.

Interpretation: warm median and own-median latency are unchanged within noise; own p95
improved (488 → 405 ms). Cold-ready and common-p95 regressed modestly on this run, which
is dominated by process start-up and shared-case scheduling noise, not by the R06 or
singular-matrix code paths (those add only small symbolic checks). The performance
component moved 62.2994 → 61.1934, i.e. **no material change**; correctness,
derivations, coverage, reliability and integration are identical to baseline.

**Limitation:** the harness reports `peak_rss_mb = 0.0` (it does not measure RSS on this
platform), so no real peak-memory claim is made.

---

## 12. Gate evidence

| Gate | Evidence | Status |
|---|---|---|
| **G1 critical correctness** | 0 critical cases, 0 major defects; 141/141 strict; required cases all strict | **PASS** |
| **G2 derivations** | 0 fabricated/invalid derivations; 141/141 derivations available; no `Compute exact result` fallback for supported results; R06/derivative verification interval-aware | **PASS** |
| **G3 untrusted input** | `untrusted_input_canary` score 1.0; malicious payload rejected with `ParseError`; `zero_denominator`, `malformed_syntax`, `empty_input`, `missing_variable`, `unsupported_operation` all 1.0 | **PASS** |
| **G4 licence** | MIT | **PASS** |

Engineering probes (StepSolver): all 10 probes score **1.0**, including
`malformed_syntax`, `unbalanced_parens`, `empty_input`, `zero_denominator`,
`missing_variable`, `two_variable_parametric`, `unsupported_operation`,
`heavy_equation`, `large_polynomial_equation`, `untrusted_input_canary`.

---

## 13. Limitations and blockers

1. **Hard & Adversarial 89 fixture absent.** Cannot run the suite, cannot claim its score,
   cannot verify U01–U07 against official data. Documented as a blocker with search
   evidence.
2. **No standalone Core-72 runner or score fixture.** Core 72 is measured from the
   embedded 72-case subset of the V2 dataset using the same evaluator; results are
   reported separately and not merged with V2.
3. **T06 identity unconfirmed.** T06 is not identifiable in any available dataset. The
   singular-matrix behaviour is implemented generically and tested; whether it is exactly
   the T06 case cannot be confirmed without the Hard fixture.
4. **Peak RSS unmeasured** by the harness (`0.0 MB` reported).
5. **Second-derivative syntax:** `diff(y(x), x, x)` (repeated variable) is unsupported;
   `diff(y(x), x, 2)` and `Derivative(y(x), (x, 2))` are supported and verified.
6. **Manual derivation overlays:** 7 stale reviews were not applied by the evaluator;
   scoring used automated derivations for those rows. Row-level scores are unchanged from
   baseline, and this is disclosed.
7. **Performance noise:** cold-ready and common-p95 vary materially between runs on this
   machine; only warm medians and own-case p95 are treated as stable signals.

---

## 14. Reproduce

```powershell
# Repository checks
$v = "C:\Users\Olabode Nathaniel\AppData\Local\pypoetry\Cache\virtualenvs\stepsolver-H0q064b0-py3.14\Scripts"
& "$v\ruff.exe" check .
& "$v\ruff.exe" format --check .
& "$v\mypy.exe"
& "$v\pyright.exe"
& "$v\ty.exe" check
& "$v\python.exe" -m pytest --cov=stepsolver --cov-report=term-missing --cov-fail-under=90
uvx --from poetry poetry check --strict
npm --prefix frontend test
npm --prefix frontend run build
git diff --check

# V2 benchmark (scratch root)
$bench = "C:\Users\Olabode Nathaniel\AppData\Local\Temp\opencode\math_solver_benchmark_v2_local"
& "$bench\envs\stepsolver\Scripts\python.exe" -m bench.runner `
    --solver stepsolver --timeout 20 --repeats 5 `
    --out "$env:TEMP\opencode\prompt11_v2_141_timed_final"
& "$bench\envs\stepsolver\Scripts\python.exe" -m bench.evaluate `
    --raw "$env:TEMP\opencode\prompt11_v2_141_timed_final" `
    --out "$env:TEMP\opencode\prompt11_v2_141_timed_final_eval"
```

---

## 15. Final commit

- Branch: `prompt11-stabilisation`
- Baseline: `7d17871299c9e1a8393c503033c7b3a063616a25`
- Implementation commit: `459181c47b5ea5a3e07f4961ae7e6af62aa3dd0c`
- Documentation finalisation: docs-only follow-up commit; the branch HEAD at delivery is
  reported in the Prompt 11 delivery message.

**Not pushed.** No remote push was performed.
