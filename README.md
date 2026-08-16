<div align="center">

# ∬ StepSolver

### A typed symbolic solver with worked steps.

</div>

> [!WARNING]
> This repository was written entirely by language models under human direction. Treat it as experimental software and review it accordingly.

<div align="center">

[Open the solver](https://davidegalilei.github.io/stepsolver/) · [Run it locally](#run-it) · [Syntax](#syntax)

[![Python 3.12–3.14](https://img.shields.io/badge/Python-3.12%E2%80%933.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-managed-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![types: strict](https://img.shields.io/badge/types-strict-2F855A)](#checks)
[![coverage: 90%+](https://img.shields.io/badge/coverage-90%25%2B-7C3AED)](#checks)
[![License: MIT](https://img.shields.io/badge/License-MIT-2F855A.svg)](LICENSE)

</div>

StepSolver accepts ASCII or visual math, parses it into its own typed AST, and returns checked solution steps. It is most useful today for single-variable calculus and algebra.

<p align="center">
  <img src="docs/assets/solver-example.png" width="920" alt="StepSolver solving the integral of x times e to the x using integration by parts">
</p>

## Run it

StepSolver supports Python 3.12 through 3.14 and uses Poetry.

```console
git clone https://github.com/DavideGalilei/stepsolver.git
cd stepsolver
poetry install
```

Start the graphical editor:

```console
poetry run stepsolver-web
```

Then open `http://127.0.0.1:8000`.

Or use the CLI:

```console
poetry run stepsolver "solve(x^2 - 4 = 0, x)"
poetry run stepsolver "limit(sin(3*x)/x, x, 0)"
poetry run stepsolver "integrate(1/(sqrt(x)*(x+1)), x)"
```

## Python API

```python
from stepsolver import ExactResult, Solver

result = Solver().solve("integrate(sin(x), x, 0, pi)")

if isinstance(result, ExactResult):
    for step in result.steps:
        print(step.rule)
```

Results and steps are immutable typed objects. Exact, divergent, and unsolved outcomes are separate result types.

## Syntax

ASCII input uses explicit operators. Write `2*x`, not `2x`. Constants include `pi`, `e`, `i`, and `oo`.

| Area | Operations |
| --- | --- |
| Algebra | `simplify`, `expand`, `factor`, `solve`, `solve_inequality` |
| Calculus | `diff`, `integrate`, `limit`, `series` |
| Sums and products | `sum`, `product` |
| Matrices | `matrix`, `det`, `inverse`, `rank`, `rref`, `eigenvalues` |
| Transforms | `laplace`, `inverse_laplace`, `fourier`, `inverse_fourier` |
| Differential equations and recurrences | `dsolve`, `rsolve` |
| Number theory | `gcd`, `lcm`, `is_prime`, `prime_factors`, `binomial` |
| Numerical work | `numeric` |

Contour integrals use an explicit parameterized path:

```text
contour_integrate(1/z, z, exp(i*t), t, 0, 2*pi)
```

Natural-language input and implicit multiplication are not supported.

## How it is put together

- The ASCII and MathJSON parsers produce the same custom AST.
- SymPy performs symbolic computation behind an adapter.
- Derivation strategies turn results into named, verified steps.
- The CLI and web app render the same result model as ASCII or LaTeX.

SymPy objects do not leak into the public API. If the backend leaves an operation unevaluated, StepSolver reports that instead of presenting it as an answer.

## Checks

```console
poetry run ruff check .
poetry run mypy src tests
poetry run pyright
poetry run ty check
poetry run pytest --cov=stepsolver --cov-fail-under=90
poetry check --strict
```

## Status

This is early-stage research software, not a replacement for a mature computer algebra system. Some supported operations still lack good student-facing derivations; those cases are tracked in the tests.

## License

StepSolver is available under the [MIT License](LICENSE).
