"""Architecture contracts for cohesive derivation modules and stable facades."""

from pathlib import Path

from stepsolver.derivation.definite_integrals import (
    derive_definite_integral,
    derive_dirichlet_integral,
    derive_improper_integral,
)
from stepsolver.derivation.definite_integrals_basic import (
    derive_definite_integral as derive_definite_integral_impl,
)
from stepsolver.derivation.definite_integrals_dirichlet import (
    derive_dirichlet_integral as derive_dirichlet_integral_impl,
)
from stepsolver.derivation.definite_integrals_improper import (
    derive_improper_integral as derive_improper_integral_impl,
)
from stepsolver.derivation.sums_convergence import CONVERGENCE_SUM_STRATEGIES
from stepsolver.derivation.sums_factorial import FACTORIAL_SUM_STRATEGIES
from stepsolver.derivation.sums_series import CLOSED_FORM_SUM_STRATEGIES

_DERIVATION_DIRECTORY = Path(__file__).parents[1] / "src" / "stepsolver" / "derivation"


def test_definite_integral_facade_preserves_the_public_functions() -> None:
    """Callers should remain insulated from the implementation-module split."""
    assert derive_definite_integral is derive_definite_integral_impl
    assert derive_improper_integral is derive_improper_integral_impl
    assert derive_dirichlet_integral is derive_dirichlet_integral_impl


def test_sum_strategy_registries_are_ordered_and_nonoverlapping() -> None:
    """Each extracted summation family should own a unique set of strategies."""
    registries = (
        FACTORIAL_SUM_STRATEGIES,
        CLOSED_FORM_SUM_STRATEGIES,
        CONVERGENCE_SUM_STRATEGIES,
    )
    strategies = tuple(strategy for registry in registries for strategy in registry)
    assert all(registry for registry in registries)
    assert all(callable(strategy) for strategy in strategies)
    assert len({strategy.__name__ for strategy in strategies}) == len(strategies)


def test_former_monoliths_remain_small_dispatch_or_facade_modules() -> None:
    """High-level modules should coordinate cohesive implementations, not absorb them again."""
    line_budgets = {
        "definite_integrals.py": 50,
        "equations.py": 200,
        "limits.py": 600,
        "sums.py": 300,
    }
    for filename, maximum_lines in line_budgets.items():
        source = (_DERIVATION_DIRECTORY / filename).read_text(encoding="utf8")
        assert len(source.splitlines()) <= maximum_lines, filename
