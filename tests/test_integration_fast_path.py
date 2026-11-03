"""Fast integration rules must not fall back to SymPy's integration search."""

import pytest
import sympy as sp

from stepsolver import ExactResult, Solver, format_ascii


def test_straightforward_integrals_use_direct_antiderivative_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G01-G03 should execute and derive steps without ``sp.integrate``."""
    solver = Solver()

    def fail_integrate(*_args: object, **_kwargs: object) -> object:
        message = "the direct antiderivative path should not call sp.integrate"
        raise AssertionError(message)

    monkeypatch.setattr(sp, "integrate", fail_integrate)
    expected = {
        "integrate(3*x^2,x)": "x ^ 3 + C",
        "integrate(2*x+5,x)": "x ^ 2 + 5 * x + C",
        "integrate(cos(x),x)": "sin(x) + C",
        "integrate(sin(3*x),x)": "-cos(3 * x) / 3 + C",
        "integrate(exp(2*x),x)": "exp(2 * x) / 2 + C",
    }
    for query, result_text in expected.items():
        result = solver.solve(query)
        assert isinstance(result, ExactResult)
        assert format_ascii(result).rstrip().endswith(f"Result: {result_text}")
