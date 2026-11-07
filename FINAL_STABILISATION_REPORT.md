# Final Stabilisation and Release-Gate Report

**Repository:** `Jayfraggs/stepsolver`
**Starting commit:** `d965b46ee2fc9a3c0e742cdcdd4b1a4de89c9b3d` (`origin/main`)
**Final code commit (benchmarked):** `ef474f6ada961a23733f84535ba7aebcf9ae8fd6`
**Branch:** `final-stabilisation`, merged into `main`
**Release decision:** **READY**

This report covers the narrowly scoped final stabilisation pass: real/complex domain propagation,
public result serialization, native `nsolve` and `dsolve` delivery, the Hard-suite real-log
antiderivative judge, structured singular-matrix results, and exhaustive regression protection.
It does not introduce a new solver, router, or heavy dependency, and it does not merge the three
benchmark suites into one score.

## 1. Commits and repository state

| Item | SHA / value |
|---|---|
| Starting commit (`main` = `origin/main`) | `d965b46ee2fc9a3c0e742cdcdd4b1a4de89c9b3d` |
| Final code commit under test | `ef474f6ada961a23733f84535ba7aebcf9ae8fd6` |
| Report and evidence commit | `60337329d216601f0d6ca9a4449ae49987db2ad1` |
| Merge target | `main` (fast-forward from `final-stabilisation`) |

**SHA circularity note.** A report cannot contain the SHA of the commit that contains it. The
code tree that produced every benchmark row in this report is `ef474f6`. The report and raw
evidence were committed as `6033732`, a child of `ef474f6`; a final metadata-only commit records
that report SHA. The repository head at handoff is the metadata commit on top of `6033732`, and
its SHA is reported in the handoff message.

No source file was changed after `ef474f6` for the benchmark runs. The installed `stepsolver`
package in both isolated benchmark environments was verified byte-for-byte against all 65 Python
files in `ef474f6` (see section 10.5).

## 2. Working-tree status and changed files

### Before

At `d965b46` the tracked tree was clean. The three suite baselines in the prompt were produced
from that commit.

### During the pass

The only working-tree changes were:

| File | Change |
|---|---|
| `src/stepsolver/results.py` | Added optional `determinant`, `rank`, and `size` to `SingularMatrixResult`. |
| `src/stepsolver/sympy_backend.py` | Populates those fields when a singular inverse is detected. |
| `Benchmarking tests/math_solver_benchmark_v2.ipynb` | Replaced the stale embedded worker with the corrected worker (typed results, domain forwarding, native `dsolve`). |
| `Benchmarking tests/math_solver_hard_suite_(claude)_colab.ipynb` | Added the maintained Hard harness with the corrected worker and real-domain judge. |
| `tests/test_result_serialization.py` | New exhaustive public result/value serialization regressions. |
| `docs/benchmarks/final_stabilisation/` | Raw before/after benchmark evidence and summaries. |
| `FINAL_STABILISATION_REPORT.md` | This report. |

No generated cache, environment, or candidate checkout is committed.

### After

All source, harness, test, report, and evidence changes are committed. The working tree is clean
apart from ignored local artefacts.

## 3. Root causes

### 3.1 The supplied V2 130/141 result came from a stale embedded worker, not a stale mathematical engine

The V2 notebook embeds its own snapshot of `bench/workers/stepsolver_worker.py`. The supplied run
used an older snapshot whose result-conversion layer predated several public result types and the
domain/native-ODE changes already present in the package. The installed `stepsolver` package in
that run was already computing the correct mathematical objects; the worker could not serialise
them.

This was confirmed by reproducing the exact 130/141 baseline with the old worker, then replacing
only the worker with the corrected one and rerunning against the same package. The failing IDs and
statuses matched the supplied run exactly.

| Case(s) | Worker-side root cause | Why it looked like a solver defect |
|---|---|---|
| C10, C14, L08 | `_val()` had no `UniversalSolutionValue` branch, so the result became `{"kind": "unknown"}` and the judge reported `unexpected answer form 'unknown'`. | The solver returned `UniversalSolutionValue(domain=REAL, exclusions=...)` with the correct human-readable derivation. |
| C15 | The old worker called `SOLVER.solve(q)` without forwarding `case["domain"]`; it recorded `domain_assumption_passed: False`. | The core real-domain solve path was never invoked, so the complex roots were returned for a real-domain case. |
| K03, K08, K09, K10, K11 | The worker's result conversion did not handle `NumericResult`, raising `UnexpectedResultType`. | The numerical roots, residuals, and derivation steps had already been computed correctly. |
| J05, J06 | `_build()` had no native `dsolve` branch, so the worker reported `no_api`. | The parser and solver already supported the operation; the dataset's `Derivative(...)` notation needed mechanical translation. |
| T06 | `_val()` had no `SingularMatrixResult` branch, leaking `UnexpectedResultType: SingularMatrixResult`. | The mathematical conclusion (determinant 0, rank 1 < 2) was already correct, but the typed public contract was incomplete. |

### 3.2 Hard R06: the real-domain judge was a false negative

The solver returned:

```text
-log(abs(x + 1)) / 2 + log(abs(x - 1)) / 2 + C
```

This is equivalent to `1/2 * log(abs((x - 1)/(x + 1))) + C`, whose derivative is `1/(x^2 - 1)`
on every real interval in the domain (`x != -1`, `x != 1`). The supplied judge differentiated
`log(Abs(...))` with a non-real-assumed symbol and called the equivalence checker with
`domain="complex"`. That can leave `re(x)`, `im(x)`, and `Derivative(re(x), x)` terms and produce
a spurious `wrong_value`.

The fix is general: the maintained Hard judge now differentiates with real-assumed symbols and
verifies the antiderivative by residual sampling on real intervals. It does not relax tolerances
and does not rewrite the solver's answer. Independent regression evidence already exists in
`tests/test_real_log_antiderivatives.py`:

- `test_interval_verification_rejects_an_inconsistent_sign_combination` proves
  `differentiates_on_real_intervals(correct, integrand, x)` is true and rejects a deliberately
  sign-inconsistent candidate.
- `test_r06_uses_absolute_logarithms_and_reports_both_poles` checks the returned form and both
  exclusions.
- `test_r06_payload_serialises_real_logarithms_and_constraints` checks the public payload keeps
  the absolute values and both constraints.

### 3.3 Hard T06: correct negative conclusion, incomplete public contract

The old worker reported `UnexpectedResultType: SingularMatrixResult`; the judge accepted the
error text as a negative result, which is why T06 was counted valid but manually flagged. The
result type now carries `determinant`, `rank`, and `size`, the worker serialises it, and the judge
accepts it only when `det == 0` and `0 <= rank < size`. This is structured evidence, not a
weakened error-text acceptance.

### 3.4 Hard U01-U07: per-case adapter overrides

The old worker read `case["adapter_inputs"]["stepsolver"]` and marked all seven ODE cases as
overrides. The corrected worker constructs every ODE query from `problem.ode`, `problem.func`,
and `problem.var`, mechanically translating the dataset's `Derivative(...)` notation. The seven
cases now run with `adapter_overrides=False`.

### 3.5 The public renderers had no exhaustiveness guard

`format_ascii` and `solve_payload` handled the current variants, but a future `SolveResult` or
`MathValue` variant could silently fall through to `Unsolved:`. The new test file asserts that the
public unions and the renderer registries list exactly the same types, so adding a variant fails
until every renderer handles it.

## 4. Fix summary mapped to the reported cases

| Case(s) | Before | Fix | After |
|---|---|---|---|
| C10, C14, L08 | `invalid_answer` (`unknown` form) | Worker serialises `UniversalSolutionValue` with domain and structured exclusions; judge accepts it. | `valid`, strict, `all real` (L08 preserves `x != 1`). |
| C15 | `invalid_answer` (complex roots for a real-domain case) | Worker forwards `case["domain"]` to `Solver.solve(..., domain=...)`; core real-domain path returns the empty set. | `valid`, strict, `no solution (empty set)`. |
| K03, K08, K09, K10, K11 | `runtime_error` (`UnexpectedResultType`) | Worker serialises `NumericResult` and its roots/residuals. | `valid`, strict, numeric roots shown below. |
| J05, J06 | `unsupported` (`no_api`) | Worker builds native `dsolve(<translated residual> = 0, y(x))`. | `valid`, strict, native ODE families. |
| R06 | `invalid_answer` (`wrong_value`) | Hard judge differentiates with real-assumed symbols and verifies on real intervals. | `valid`, strict; derivation 10/10. |
| T06 | `valid` but manual-review flagged (`UnexpectedResultType`) | `SingularMatrixResult` carries `det`, `rank`, `size`; worker serialises it; judge accepts structured evidence. | `valid`, strict, `singular matrix: det=0 rank=1/2`. |
| U01-U07 | `valid` but `adapter_overrides=True` | Worker constructs from `problem.ode`/`func`/`var`; legacy per-case overrides ignored. | `valid`, strict, `adapter_overrides=False`. |

Numeric roots accepted after the fix:

| Case | Equation | Guess | Accepted root |
|---|---|---:|---:|
| K03 | `cos(x) = x` | 0.7 | `[0.739085133215161]` |
| K08 | `x^3 - 2*x - 5 = 0` | 2 | `[2.09455148154233]` |
| K09 | `exp(-x) = x` | 0.5 | `[0.567143290409781]` |
| K10 | `sin(x) = 1/2` | 0.5 | `[0.523598775598299]` |
| K11 | `ln(x) = 1` | 2.5 | `[2.71828182845905]` |

## 5. Tests added and exact commands executed

### New tests

`tests/test_result_serialization.py` contains 18 tests:

| Test | Purpose |
|---|---|
| `test_solve_result_union_matches_serialization_registry` | `SolveResult` and the renderer registry list exactly the same six result types. |
| `test_math_value_union_matches_value_registry` | `MathValue` and the sample registry list exactly the same eight value types. |
| `test_every_result_type_serializes_with_its_own_status` (6 params) | Each result variant renders as itself and gets its own payload status, never `unsolved`. |
| `test_singular_matrix_result_carries_determinant_rank_and_size` | The singular result exposes the facts that justify "no inverse". |
| `test_every_math_value_renders` (8 params) | Each value variant renders without falling through to a default. |
| `test_universal_value_reports_domain_and_exclusions` | Universal values serialise domain and exclusions explicitly. |

Existing R06 regressions are in `tests/test_real_log_antiderivatives.py` (real logarithms,
nearby rational integrals, interval verification, payload constraints, G04/G08 protection).

### Commands

Python (project virtualenv, Python 3.14.4):

```text
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pyright
python -m ty check
python -m pytest --cov=stepsolver --cov-report=term-missing
uvx --from poetry poetry check --strict
git diff --check
```

Frontend (`frontend/`):

```text
npm test
npm run build
```

Benchmarks (isolated benchmark environments, Python 3.12.13, SymPy 1.14.0):

```text
python -m bench.runner --solver stepsolver --timeout 20 --repeats 5
python -m bench.evaluate --raw results/raw --out results
python -m bench.engineering --out results/engineering_results.json
python -m bench.score
```

For the Hard "before" reproduction only, the original worker and judge were temporarily restored,
the nine prior-failure cases were run (`--repeats 1 --timeout 30`), the results were evaluated,
and the corrected files were restored. The restored files were verified byte-identical (after LF
normalisation) to the committed notebook payloads, and a 9/9 smoke rerun passed.

## 6. Full test-suite results

| Gate | Result |
|---|---|
| Python tests | **849 passed** (120.49 s) |
| Coverage | **90.30%** (threshold 90%) |
| Ruff lint | All checks passed |
| Ruff format | 112 files already formatted |
| mypy (strict) | Success: no issues found in 104 source files |
| Pyright | 0 errors, 56 SymPy-stub `reportMissingModuleSource` warnings |
| ty | All checks passed |
| Frontend tests | **26/26 passed** |
| Frontend production build | Passed |
| Poetry strict check | `All set!` |
| `git diff --check` | Clean |

## 7. Benchmark results (three separate suites)

The three suites are reported separately and are never merged or averaged.

### 7.1 Core 72 (embedded subset of the V2 dataset)

Core 72 is a 72-case subset of the 141-case V2 run. It is **not** a separately gated run; the
hard gates below are computed by the V2 harness over all 141 cases.

| Metric | Before (`d965b46`) | After (`ef474f6`) |
|---|---:|---:|
| Cases | 72 | 72 |
| Strict correct | 70/72 | **72/72** |
| Covered | 70/72 | **72/72** |
| Status counts | 70 valid, 1 invalid, 1 runtime error | **72 valid** |
| Derivations available | 72/72 | 72/72 |
| Derivation mean (final) | 9.1831 | **9.2639** |
| Derivation mean (automated) | 9.1549 | 9.2361 |
| Domain-sensitive strict | 20/21 | **21/21** |
| Adapter overrides | 2 (J01, J02) | **0** |
| Warm median (own) | not measured | **10.493 ms** |
| Warm p95 (own) | not measured | 93.183 ms |
| Hard gates | n/a (subset of V2 run) | **n/a (V2 G1-G4 PASS)** |

Evidence: `docs/benchmarks/final_stabilisation/core_72_summary.json`,
`core_72_before_summary.json`, `v2_141_rows.json`.

### 7.2 V2 Stress 141

| Metric | Before (`d965b46`, reproduced) | After (`ef474f6`) |
|---|---:|---:|
| Cases | 141 | 141 |
| Strict correct | 130/141 | **141/141** |
| Covered | 131/141 | **141/141** |
| Status counts | 130 valid, 4 invalid, 5 runtime errors, 2 unsupported | **141 valid** |
| Derivations available | 139/141 | **141/141** |
| Derivation mean (final) | 9.1493 | **9.3262** |
| Derivation mean (automated) | 9.1343 | 9.3121 |
| Domain-sensitive strict | 43/47 | **47/47** |
| Adapter overrides | 2/141 (J01, J02) | **0/141** |
| Overall score | 92.13 (supplied) | **96.1979** |
| Correctness | 100.0 | **100.0** |
| Derivations component | - | 93.26 |
| Coverage | 100.0 | **100.0** |
| Reliability | 100.0 | **100.0** |
| Integration | - | 85.71 |
| Performance | - | 78.67 |
| Hard gates | G1 FAIL (C15); G2/G3/G4 PASS | **G1-G4 PASS**, eligible **YES** |
| Warm median (own) | not measured | **12.910 ms** |
| Warm p95 (own) | not measured | 126.604 ms |
| Warm median (82 common cases) | not measured | 19.978 ms |
| Warm p95 (82 common cases) | not measured | 84.433 ms |
| Cold start / first solve | not measured | 623.114 ms / 44.833 ms |
| Timeouts / crashes | 0 / 0 | **0 / 0** |

All 12 categories are 100% strict and covered: A 13/13, B 13/13, C 16/16, D 11/11, E 13/13,
F 12/12, G 13/13, H 9/9, I 9/9, J 8/8, K 12/12, L 12/12. Category E (trigonometric) remains
13/13.

The score is stable across the harness's sensitivity variants: primary 96.1979, value-only
96.1979, value-ignoring-form 96.1979, availability-penalised 96.1979, automated-only 96.1553,
own-supported-latency 96.2098. Reliability R1-R5 are all 1.0.

Evidence: `docs/benchmarks/final_stabilisation/v2_141_summary.json`,
`v2_141_before_summary.json`, `v2_141_rows.json`, `v2_141_raw_replies.json`,
`v2_141_raw_results.json`, `v2_141_scores.json`, `v2_141_engineering.json`.

### 7.3 Hard & Adversarial 89

| Metric | Before (`d965b46`, supplied) | After (`ef474f6`) |
|---|---:|---:|
| Cases | 89 | 89 |
| Strict correct | 88/89 | **89/89** |
| Covered | 89/89 | **89/89** |
| Status counts | 88 valid, 1 invalid (R06); T06 manual-review flagged | **89 valid** |
| Derivations available | 89/89 | **89/89** |
| Derivation mean (final) | 9.23 | **9.2921** |
| Domain-sensitive strict | - | **13/13** |
| Adapter overrides | 7/89 (U01-U07) | **0/89** |
| Overall score | 96.09 | **96.7308** |
| Correctness | 100.0 | **100.0** |
| Derivations component | - | 92.92 |
| Coverage | 100.0 | **100.0** |
| Honesty | 100.0 | **100.0** |
| Reliability | 100.0 | **100.0** |
| Integration | - | 85.71 |
| Performance | - | 59.54 |
| Hard gates | G1 FAIL (R06); G2/G3/G4 PASS | **G1-G4 PASS**, eligible **YES** |
| Warm median (own) | not measured | **46.147 ms** |
| Warm p95 (own) | not measured | 460.757 ms |
| Cold start / first solve | not measured | 738.164 ms / 206.233 ms |
| Timeouts / crashes | 0 / 0 | **0 / 0** |

Tier results are 100% strict and covered: tier 1 37/37, tier 2 42/42, tier 3 (adversarial) 10/10.
All 10 categories are 100% strict and covered: M 8/8, N 11/11, O 9/9, P 5/5, Q 5/5, R 15/15,
S 17/17, T 9/9, U 7/7, V 3/3. The five trigonometric cases P01-P05 remain 5/5.

The score is identical across all eight sensitivity variants (96.7308), including
adversarial-cases-only and automated-only derivations. Reliability R1-R5 are all 1.0.

Evidence: `docs/benchmarks/final_stabilisation/hard_89_summary.json`,
`hard_89_rows.json`, `hard_89_raw_replies.json`, `hard_89_raw_results.json`,
`hard_89_scores.json`, `hard_89_engineering.json`, `hard_89_evidence.json`,
`hard_89_environment.txt`. The prior-failure subset is preserved in
`hard_89_before_prior_failures_*`.

## 8. Before/after comparison against `d965b46`

The status and gate comparisons are valid because the suites, cases, judge rules, and hard-gate
definitions are unchanged. The local after-runs use Windows/Python 3.12.13/SymPy 1.14.0; the
supplied baseline hardware/configuration is not byte-identical, so score and latency differences
are directional rather than a controlled speed comparison.

| Suite | Before | After | Change |
|---|---:|---:|---:|
| Core 72 strict | 70/72 | 72/72 | +2 |
| Core 72 covered | 70/72 | 72/72 | +2 |
| V2 141 strict | 130/141 | 141/141 | +11 |
| V2 141 covered | 131/141 | 141/141 | +10 |
| V2 141 derivation mean | 9.1493 | 9.3262 | +0.1769 |
| V2 141 overrides | 2 | 0 | -2 |
| V2 141 score | 92.13 | 96.1979 | +4.07 |
| Hard 89 strict | 88/89 | 89/89 | +1 |
| Hard 89 covered | 89/89 | 89/89 | 0 |
| Hard 89 overrides | 7 | 0 | -7 |
| Hard 89 score | 96.09 | 96.7308 | +0.64 |

The V2 "before" row is a local reproduction of the supplied baseline with the stale embedded
worker: 130 valid, 4 invalid, 5 runtime errors, 2 unsupported, 139/141 derivations available,
mean 9.1493, failing IDs C10, C14, C15, J05, J06, K03, K08, K09, K10, K11, L08. This matches the
supplied baseline exactly. The Hard "before" full-suite row is the supplied baseline; the local
9-case reproduction confirms R06 `wrong_value`, T06 `error_signalled_negative_result` with
`UnexpectedResultType`, and U01-U07 with `adapter_overrides=True`.

## 9. Case-level table for every prior failure

| Case | Suite | Before status | Before detail | After status | After actual |
|---|---|---|---|---|---|
| C10 | V2 | `invalid_answer` | `unexpected answer form 'unknown'` | `valid` (strict) | `all real` |
| C14 | V2 | `invalid_answer` | `unexpected answer form 'unknown'` | `valid` (strict) | `all real` |
| L08 | V2 | `invalid_answer` | `unexpected answer form 'unknown'` | `valid` (strict) | `all real`; exclusion `x != 1` preserved |
| C15 | V2 | `invalid_answer` | `returned solutions where empty expected` | `valid` (strict) | `no solution (empty set)` |
| K03 | V2 | `runtime_error` | `UnexpectedResultType` | `valid` (strict) | `[0.739085133215161]` |
| K08 | V2 | `runtime_error` | `UnexpectedResultType` | `valid` (strict) | `[2.09455148154233]` |
| K09 | V2 | `runtime_error` | `UnexpectedResultType` | `valid` (strict) | `[0.567143290409781]` |
| K10 | V2 | `runtime_error` | `UnexpectedResultType` | `valid` (strict) | `[0.523598775598299]` |
| K11 | V2 | `runtime_error` | `UnexpectedResultType` | `valid` (strict) | `[2.71828182845905]` |
| J05 | V2 | `unsupported` | `no_api` | `valid` (strict) | `Eq(y(x), C1 * exp(-2 * x))` |
| J06 | V2 | `unsupported` | `no_api` | `valid` (strict) | `Eq(y(x), (C1 + C2 * exp(x)) * exp(x))` |
| R06 | Hard | `invalid_answer` | `wrong_value` | `valid` (strict), derivation 10/10 | `-log(abs(x + 1)) / 2 + log(abs(x - 1)) / 2 + C` |
| T06 | Hard | `valid`, manual-review flagged | `error_signalled_negative_result`; `UnexpectedResultType: SingularMatrixResult` | `valid` (strict), no manual flag | `singular matrix: det=0 rank=1/2` |
| U01 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), C1 * exp(-x) + x - 1)` |
| U02 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), (C1 + C2 * exp(x)) * exp(x))` |
| U03 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), (C1 + C2 * x) * exp(x))` |
| U04 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), (C1 * sin(2 * x) + C2 * cos(2 * x)) * exp(-x))` |
| U05 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), C1 * exp(x ^ 2 / 2))` |
| U06 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), C1 * sin(x) + C2 * cos(x) + x)` |
| U07 | Hard | `valid`, override | `adapter_overrides=True` | `valid` (strict), no override | `Eq(y(x), -1 / (C1 + x))` |

J01 and J02 were also converted from legacy `dsolve` override inputs to native construction; they
were already strict before, so they do not appear in the failure table.

## 10. Remaining judge limitations and manual-review flags

### 10.1 V2 manual derivation overlays

The V2 derivation score uses 14 applied manual overlays out of 39 total review entries; 7 entries
are stale and were not applied (A07, B06, C07, D04, I03, J01, L02). The overlays affect only the
derivation-quality score, never correctness, coverage, or hard gates. The automated-only variant
scores 96.1553, so the overlays are not masking an automated failure. The Hard suite has **0**
manual overlays.

### 10.2 R06 judge

The Hard judge is maintained in this workspace and was corrected generally:
real-assumed symbols for `log(Abs(...))` differentiation and residual sampling on real intervals.
No tolerance was widened. The independent `differentiates_on_real_intervals` test rejects a
deliberately inconsistent candidate, so the verification is not a blanket acceptance.

### 10.3 T06 structured singular acceptance

The judge accepts the singular outcome only when `determinant == 0` and `0 <= rank < size`. The
worker's answer for T06 is `det=0 rank=1/2`, matching the matrix `[[1, 2], [2, 4]]`.

### 10.4 Measurement limitations

- `peak_rss_mb` is reported as `0.0` by the harness. No memory claim is made.
- The Hard performance component (59.54) is low relative to the other components, but it is not a
  hard gate and no timeout or crash occurred. The p95 is dominated by a small number of heavy
  cases; the median is 46.147 ms.
- V2 performance uses 82 cases common to all engines; Hard uses all 89.
- Before/after latency and score comparisons are directional because the supplied baseline
  hardware is not identical to the local Windows run.

### 10.5 Benchmark environment provenance

The isolated benchmark environments were originally built from commit `89b16c6` during the V2
harness setup, then updated in place with the final package source. All 65 `.py` files in both
`envs/stepsolver` trees were compared byte-for-byte against `git show ef474f6:src/stepsolver/...`
with **0 mismatches**. The V2 raw rows therefore carry the environment's original metadata string
`89b16c643b034e716002622068570a47f04d2a91`, while the code bytes are `ef474f6`. The Hard evidence
file records `ef474f6ada961a23733f84535ba7aebcf9ae8fd6` as the code commit under test. This is a
provenance note, not a correctness caveat.

### 10.6 Licence and dependencies

MIT licence (permissive), 7 runtime dependencies, Python `>=3.12,<3.15`. G4 PASS in both suites.

## 11. Preservation confirmation

| Requirement | Evidence |
|---|---|
| Safe parser allow-list unchanged | No parser code was modified. G3 canary score 1.0; payload rejected with `ParseError`. |
| No `eval`/`exec` or arbitrary Python evaluation | No such code was added; the worker only translates `Derivative(...)` to `diff(...)` mechanically. |
| No case-ID branches or hard-coded answers in core | Core changes are the generic `SingularMatrixResult` fields and their population. Case IDs appear only in tests/evidence. |
| No judge weakening | G2 PASS (0 fabricated/invalid derivations); R06 tolerances unchanged; T06 requires structured determinant/rank/size. |
| Trigonometric cases preserved | V2 category E 13/13; Hard P01-P05 5/5; K10 numeric trigonometric root accepted. |
| Exact arithmetic preserved | Correctness component 100.0 in both V2 and Hard; all exact forms/domain checks pass. |
| Derivation integrity preserved | 0 opaque steps, 0 invalid steps, 0 substantive-zero derivations in both suites; 141/141 and 89/89 derivations available. |
| Suites kept separate | Core 72, V2 141, and Hard 89 are reported in separate tables; no combined or averaged score. |

## 12. Final release decision

**READY.**

Justification from hard-gate evidence:

- **Core 72:** 72/72 strict, 72/72 covered, 0 invalid/errors/unsupported, 72/72 derivations,
  mean 9.2639, 0 overrides.
- **V2 141:** 141/141 strict, 141/141 covered, 0 invalid/errors/unsupported, 141/141 derivations,
  mean 9.3262, 0 overrides, score 96.1979, **G1-G4 PASS**, eligible **YES**.
- **Hard 89:** 89/89 strict, 89/89 covered, 0 invalid/errors/unsupported, 89/89 derivations,
  mean 9.2921, 0 overrides, score 96.7308, **G1-G4 PASS**, eligible **YES**.
- **Quality gates:** 849 Python tests at 90.30% coverage, frontend 26/26 tests plus production
  build, Ruff, mypy, Pyright, ty, Poetry strict check, and `git diff --check` all pass.

No known solver, adapter, judge, or safety defect remains. The only recorded limitations are the
measurement caveats in section 10 (manual derivation overlays, unavailable peak RSS, cross-hardware
latency comparability, and the benchmark-environment metadata string), none of which affect the
release gate.

---

## Evidence index

All paths are relative to the repository root.

| File | Contents |
|---|---|
| `docs/benchmarks/final_stabilisation/core_72_summary.json` | Core 72 after summary. |
| `docs/benchmarks/final_stabilisation/core_72_before_summary.json` | Core 72 reproduced before summary. |
| `docs/benchmarks/final_stabilisation/v2_141_summary.json` | V2 141 after summary. |
| `docs/benchmarks/final_stabilisation/v2_141_before_summary.json` | V2 141 reproduced before summary. |
| `docs/benchmarks/final_stabilisation/v2_141_rows.json` | Compact after rows (141). |
| `docs/benchmarks/final_stabilisation/v2_141_before_rows.json` | Compact before rows (141). |
| `docs/benchmarks/final_stabilisation/v2_141_raw_replies.json` | Raw after worker replies. |
| `docs/benchmarks/final_stabilisation/v2_141_before_raw_replies.json` | Raw before worker replies. |
| `docs/benchmarks/final_stabilisation/v2_141_raw_results.json` | Raw after evaluated rows. |
| `docs/benchmarks/final_stabilisation/v2_141_before_raw_results.json` | Raw before evaluated rows. |
| `docs/benchmarks/final_stabilisation/v2_141_scores.json` | V2 score with gates. |
| `docs/benchmarks/final_stabilisation/v2_141_engineering.json` | V2 engineering probes. |
| `docs/benchmarks/final_stabilisation/hard_89_summary.json` | Hard 89 after summary. |
| `docs/benchmarks/final_stabilisation/hard_89_rows.json` | Compact after rows (89). |
| `docs/benchmarks/final_stabilisation/hard_89_raw_replies.json` | Raw after worker replies. |
| `docs/benchmarks/final_stabilisation/hard_89_raw_results.json` | Raw after evaluated rows. |
| `docs/benchmarks/final_stabilisation/hard_89_scores.json` | Hard score with gates. |
| `docs/benchmarks/final_stabilisation/hard_89_engineering.json` | Hard engineering probes. |
| `docs/benchmarks/final_stabilisation/hard_89_evidence.json` | Hard setup evidence (licence, history, deps, tests). |
| `docs/benchmarks/final_stabilisation/hard_89_environment.txt` | Hard environment record. |
| `docs/benchmarks/final_stabilisation/hard_89_before_prior_failures_summary.json` | Nine prior-failure cases before fixes. |
| `docs/benchmarks/final_stabilisation/hard_89_before_prior_failures_rows.json` | Compact before rows (9). |
| `docs/benchmarks/final_stabilisation/hard_89_before_prior_failures_raw_replies.json` | Raw before replies (9). |
| `docs/benchmarks/final_stabilisation/hard_89_before_prior_failures_raw_results.json` | Raw before evaluated rows (9). |
