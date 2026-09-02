import pytest
from click.testing import CliRunner

import lionz.lionz as lionz_cli


def test_cli_rejects_a_missing_input_before_runtime_initialization(
    tmp_path,
    monkeypatch,
):
    missing_directory = tmp_path / f"missing-input-{'x' * 120}"

    def fail_if_called(*args, **kwargs):
        pytest.fail("runtime initialization must not run for an invalid input path")

    monkeypatch.setattr(lionz_cli, "execute_cli", fail_if_called)
    monkeypatch.setattr(lionz_cli.system, "check_device", fail_if_called)

    result = CliRunner().invoke(
        lionz_cli.main,
        ["-d", str(missing_directory), "-m", "psma"],
        color=True,
        terminal_width=40,
    )

    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert "Invalid value" in result.output
    assert list(tmp_path.rglob("*.log")) == []
