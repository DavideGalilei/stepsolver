# Hard & Adversarial 89-case suite — BLOCKED

**Status:** ⛔ not run. No score is claimed.

## What was required

Prompt 11 requires evaluation against three suites reported separately:

1. Core — 72 cases
2. Hard & Adversarial — 89 cases
3. V2 Stress — 141 cases

## What exists

| Location | Contents |
|---|---|
| `Benchmarking tests/math_solver_benchmark_colab.ipynb` | Original benchmark: 72 core case IDs (A01–L04) |
| `Benchmarking tests/math_solver_benchmark_v2.ipynb` | V2: same 72 core IDs + 69 stress cases = 141 |
| `Benchmarking tests/open_source_math_solver_benchmark_colab.ipynb` | Open-source comparison, no Hard 89 data |
| `bench/data/cases.json` (scratch benchmark root) | Exactly 141 cases (A01–L12), no Hard 89 rows |
| `bench/results/`, `bench/docs/`, `bench/candidates/`, `bench/pins/` | No 89-case fixture |

## Search performed

- Recursive filename search across the repository and all sibling directories under
  `C:\Users\Olabode Nathaniel\Code\Step Solver` for `hard`, `adversarial`, `89`, `suite`,
  `case` (excluding `node_modules`, `.git`, `site-packages`, `__pycache__`, `.ruff_cache`,
  `envs`).
- Recursive content search for `Hard 89`, `Hard-89`, `89 hard`, `U01`–`U29`,
  `adversarial 89`, `89-case`.
- Notebook source inspection: the V2 notebook's own text describes 72 core + 69 stress =
  141, not 161.

Result: **no 89-case fixture exists in the workspace or benchmark environment.**

## Consequence

The following cannot be verified against official rows:

- R06 acceptance (verified instead by targeted before/after probes and tests)
- T06 identity and acceptance (behaviour implemented generically and tested)
- U01–U07 and any Hard-specific N-/O-suite overrides
- The Hard 89 overall score

## What is claimed instead

- R06: `-log(abs(x + 1))/2 + log(abs(x - 1))/2 + C` with `x != -1`, `x != 1`;
  five nearby rational integrands verified.
- T06 class: typed `SingularMatrixResult`, `status="singular"`, determinant/rank reasoning.
- V2 Stress 141: 141/141 strict, overall 95.3241.
- Core 72 (embedded subset): 72/72 strict.

No Hard 89 pass or score is fabricated.
