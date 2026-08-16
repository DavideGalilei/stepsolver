"""Command-line interface tests."""

import sys
from io import StringIO

import pytest

from stepsolver.cli import main

_PARSE_ERROR_EXIT_CODE = 2


def test_cli_query_argument(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should solve a query supplied as an argument."""
    exit_code = main(["factor(x^2-1)"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Result: (x - 1) * (x + 1)" in captured.out
    assert not captured.err


def test_cli_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid syntax should be reported through stderr and exit code two."""
    exit_code = main(["2x"])
    captured = capsys.readouterr()
    assert exit_code == _PARSE_ERROR_EXIT_CODE
    assert "Error:" in captured.err


def test_cli_unsolved_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    """A valid but unsupported query should use exit code one."""
    exit_code = main(["integrate(x)"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Unsolved:" in captured.out


def test_cli_reads_standard_input(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should read a query from standard input when no argument is supplied."""
    monkeypatch.setattr(sys, "stdin", StringIO("gcd(21, 14)"))
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Result: 7" in captured.out
