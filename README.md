# StepSolver

StepSolver is a strictly typed symbolic mathematics library and CLI. It parses
ASCII input into its own immutable AST, delegates difficult symbolic operations
through a narrow SymPy adapter, and returns backend-independent typed results
with machine-checked transformation records.

## Install

```console
poetry install
```

## Python API

```python
from stepsolver import Solver, format_ascii

result = Solver().solve("integrate(sin(x), x, 0, pi)")
print(format_ascii(result))
```

## CLI

```console
poetry run stepsolver "solve(x^2 - 4 = 0, x)"
poetry run stepsolver "contour_integrate(1/z, z, exp(i*t), t, 0, 2*pi)"
```

When no argument is supplied, the CLI reads one query from standard input.

## Web frontend

Start the FastAPI server and open `http://127.0.0.1:8000`:

```console
poetry run stepsolver-web
```

The browser frontend provides one universal graphical equation editor and an
always-visible palette for fractions, roots, relations, calculus notation,
functions, and constants. Enter the complete problem as conventional
two-dimensional notation: an equation is solved, an integral is integrated,
a derivative is differentiated, and a bare expression is simplified. No
operation dropdown or separate variable/bounds form is needed. Answers and
steps are rendered as semantic mathematical notation. The normalized ASCII
solver input remains available in an expandable diagnostics panel.

Server settings can be changed through environment variables:

```console
STEPSOLVER_HOST=0.0.0.0 STEPSOLVER_PORT=8080 STEPSOLVER_RELOAD=true \
  poetry run stepsolver-web
```

The graphical editor uses `POST /api/solve` with a body such as:

```json
{
  "latex": "\\int_0^\\pi \\sin(x)\\,\\mathrm{d}x",
  "math_json": [
    "Integrate",
    ["Sin", "x"],
    ["Tuple", "x", 0, "Pi"]
  ]
}
```

The browser derives semantic MathJSON from the visual field. The server
validates that tree, converts it into StepSolver's custom AST, and infers the
operation from the outer notation. The LaTeX string is display context only;
it is not evaluated by the backend.

## Syntax

- Explicit operators: `+`, `-`, `*`, `/`, `^`, and `!`.
- Relations: `=`, `!=`, `<`, `<=`, `>`, and `>=`.
- Exact constants: `pi`, `e`, `i`, and `oo`.
- Explicit multiplication is required: write `2*x`, not `2x`.
- Top-level function syntax selects an operation. A bare expression is
  simplified.

Common operations include `simplify`, `expand`, `factor`, `solve`,
`solve_inequality`, `diff`, `integrate`, `limit`, `series`, `sum`, `product`,
`matrix`, `det`, `inverse`, `rank`, `rref`, `eigenvalues`, `dsolve`, `rsolve`,
`laplace`, `fourier`, `gcd`, `lcm`, `is_prime`, `prime_factors`, `binomial`,
`permutations`, `combinations`, and `numeric`.

Parameterized contour integrals use:

```text
contour_integrate(integrand, complex_variable, path, parameter, lower, upper)
```

For example, the positively oriented unit circle is `exp(i*t)` for
`0 <= t <= 2*pi`.

StepSolver does not execute Python input and does not use natural-language or
implicit-multiplication parsing. Valid problems without a verified supported
result return a typed `UnsolvedResult`.
