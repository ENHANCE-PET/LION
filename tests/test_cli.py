from click.testing import CliRunner

from lionz.lionz import main


def test_cli_rejects_a_missing_input_directory(tmp_path):
    missing_directory = tmp_path / "missing-input"

    result = CliRunner().invoke(
        main,
        ["-d", str(missing_directory), "-m", "psma"],
    )

    assert result.exit_code == 2
    assert missing_directory.name in result.output
    assert "Invalid value for '-d' / '--main-directory'" in result.output
    assert "CUDA initialization" not in result.output
    assert "lionz-v1.0.5_" not in result.output
