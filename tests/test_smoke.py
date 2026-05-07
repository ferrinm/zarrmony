"""Smoke tests for the v0.1 scaffolding. Replace with real tests as features land."""

from importlib.metadata import version

from click.testing import CliRunner

import zarrmony
from zarrmony.cli import app


def test_version_set() -> None:
    assert zarrmony.__version__ == version("zarrmony")


def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "zarrmony" in result.output.lower()


def test_cli_subcommands_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    for cmd in ("convert", "inspect", "schema"):
        assert cmd in result.output
