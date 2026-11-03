"""The optional web entry point should fail with an actionable message."""

import importlib

import pytest

from stepsolver import web_entry


def test_missing_web_extra_explains_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base install should not emit a bare ModuleNotFoundError traceback."""

    def fake_import(_name: str) -> object:
        message = "No module named 'fastapi'"
        raise ModuleNotFoundError(message, name="fastapi")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(SystemExit, match=r"pip install 'stepsolver\[web\]'"):
        web_entry.main()


def test_unrelated_import_errors_are_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only missing web dependencies should receive the extra hint."""

    def fake_import(_name: str) -> object:
        message = "No module named 'unrelated'"
        raise ModuleNotFoundError(message, name="unrelated")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError, match="unrelated"):
        web_entry.main()
