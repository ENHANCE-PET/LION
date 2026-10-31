from unittest.mock import MagicMock

from lionz.file_utilities import set_permissions


def test_windows_executable_does_not_grant_write_access_to_other_users(tmp_path, monkeypatch):
    executable = tmp_path / "c3d.exe"
    executable.write_bytes(b"synthetic")
    calls = []
    monkeypatch.setattr("lionz.file_utilities.subprocess.check_call", lambda args, **kwargs: calls.append(args))
    set_permissions(str(executable), "windows", MagicMock())
    assert not any("Everyone:(F)" in argument for args in calls for argument in args)
