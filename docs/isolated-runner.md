# Isolated runner contract

Last updated: 2026-08-28

The isolated runner is StepSolver's narrow machine-facing entry point. It is
designed for a disposable worker with an independent wall-clock, memory, CPU,
and output limit. It does not provide a Python evaluation surface.

## Protocol

The process reads exactly one UTF-8 JSON document from standard input and writes
exactly one compact JSON document to standard output.

The only accepted request shape is:

```json
{"expression":"solve(x^2-4=0,x)"}
```

Unknown fields, non-string expressions, trailing input, invalid UTF-8, and
non-ASCII mathematics are rejected. A successful response is:

```json
{
  "ok": true,
  "solution": {
    "normalized_expression": "solve(x ^ 2 - 4 = 0, x)",
    "status": "exact",
    "result": "x = -2 or x = 2",
    "latex": "x = -2\\quad\\text{or}\\quad x = 2",
    "reason": null,
    "steps": []
  }
}
```

A rejected or unsafely large request returns a stable error code without a
traceback. Solver failures are data, not process-control messages.

## Independent limits

The library validates the parsed request before invoking SymPy:

- request: 4 KiB; expression: 2,048 ASCII characters;
- AST: 256 nodes, depth 32, 64 symbols, 64 sequence items per node, and 32
  function arguments per call;
- exact integers: 100 decimal digits in either numerator or denominator;
- numeric powers: absolute exponent at most 100;
- matrices: at most 8 by 8;
- finite sums and products: at most 10,000 terms;
- result: 64 steps, 16 notes and 16 introduced constraints per step, and 64 KiB
  after compact JSON serialization.

Limit checks calculate facts from the immutable StepSolver AST and return a
validated request. Symbolic execution receives only that validated value.

The worker supervisor must enforce the same payload and output ceilings plus an
eight-second wall deadline, one vCPU, 256 MiB RAM, a read-only root filesystem,
a bounded temporary filesystem, and no network interface. Library checks do not
replace the microVM boundary.

## Lifecycle

One worker handles one request and exits. The host supervisor never reuses a
worker after symbolic execution. It keeps at most two warm workers, admits at
most two active solves, and uses a bounded queue. A timed-out, malformed, or
disconnected worker is destroyed before replacement.

The runner logs nothing to standard output except its response. Operational
diagnostics belong on standard error and must not include the expression.
