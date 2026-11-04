# Prompt 11 benchmark evidence

Separate suites, never combined into one artificial score.

| Suite | Cases | Result |
|---|---|---|
| Core | 72 | **72/72 strict**, 72/72 covered, 72/72 derivations, mean final 9.2639 |
| Hard & Adversarial | 89 | ⛔ **BLOCKED — fixture absent** (see `hard_89_BLOCKED.md`) |
| V2 Stress | 141 | **141/141 strict**, 141/141 covered, overall **95.3241** |

## Files

| File | Contents |
|---|---|
| `core_72_summary.json` | Core 72 embedded-subset metrics and case-ID definition |
| `v2_141_rows.json` | Compact per-row extract: status, strict correctness, answer, restrictions, adapter overrides, derivation score, warm median |
| `v2_141_scores.json` | Unmodified `bench/score.py` output for the merged shadow run |
| `engineering_probes.json` | Engineering probe results (untrusted-input canary, malformed input, heavy cases) |
| `hard_89_BLOCKED.md` | Why the Hard 89 suite could not be run |

## Provenance

- **Raw V2 run:** `%TEMP%\opencode\prompt11_v2_141_timed_final\stepsolver.json`
  (`bench.runner --solver stepsolver --timeout 20 --repeats 5`)
- **Evaluated rows:** `%TEMP%\opencode\prompt11_v2_141_timed_final_eval\raw_results.json`
  (`bench.evaluate`)
- **Scores:** shadow root `%TEMP%\opencode\prompt11_shadow_score`, built by merging the new
  StepSolver rows with the unchanged SymPy Direct / SmartSolver baseline rows, then running
  the unmodified `bench/score.py`. Preserved baseline result files were not modified.

## Headline metrics

| Metric | Value |
|---|---|
| V2 strict correct | 141/141 |
| V2 covered | 141/141 |
| V2 adapter overrides | 0/141 (baseline 2/141) |
| V2 timeouts / crashes | 0 / 0 |
| V2 correctness | 100.0 |
| V2 derivations | 93.2624 |
| V2 coverage | 100.0 |
| V2 reliability | 100.0 |
| V2 integration | 85.7143 |
| V2 performance | 61.1934 |
| V2 overall | 95.3241 |
| Core warm median | 33.461 ms |
| V2 warm median (common) | 61.98 ms |
| V2 own median | 40.88 ms |
| V2 own p95 | 405.37 ms |

## Hard gates

| Gate | Status |
|---|---|
| G1 critical correctness | PASS |
| G2 fabricated/invalid derivations | PASS |
| G3 unsafe execution | PASS — canary score 1.0, payload rejected with `ParseError` |
| G4 licence | PASS — MIT |

## Caveats

- The Hard 89 fixture is not present anywhere in the workspace; no score is claimed.
- The harness reports `peak_rss_mb = 0.0`; no memory claim is made.
- 7 stale manual derivation reviews were not applied by the evaluator; those rows were
  scored automatically. Row-level derivation totals are unchanged from baseline.
- Cold-ready and common-p95 timings vary materially between runs on this machine; only
  warm medians and own-case p95 are treated as stable signals.
