"""Downloads are mocked; archives and payloads are synthetic."""
import hashlib
import io
from unittest.mock import MagicMock
import zipfile

import pytest

from lionz import models


def archive_bytes(name, payload=b"synthetic"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


def download(tmp_path, monkeypatch, data, digest=None, fail_download=False):
    model = models.Model.__new__(models.Model)
    model.model_identifier = "synthetic"
    model.folder_name = "DatasetSynthetic"
    model.base_directory = str(tmp_path)
    model.directory = str(tmp_path / model.folder_name)
    model.url = "https://example.invalid/synthetic.zip"
    model.sha256 = digest or hashlib.sha256(data).hexdigest()
    response = MagicMock(status_code=200, headers={"Content-Length": str(len(data))})
    response.iter_content.return_value = [data]
    response.__enter__.return_value = response
    if fail_download:
        response.raise_for_status.side_effect = models.requests.ConnectionError("offline")
    monkeypatch.setattr(models.requests, "get", lambda *args, **kwargs: response)
    model._Model__download(MagicMock())


def test_model_download_rejects_files_outside_model_folder(tmp_path, monkeypatch):
    victim = tmp_path / "victim.txt"
    victim.write_text("existing")
    data = archive_bytes("victim.txt", b"overwritten")
    with pytest.raises(ValueError):
        download(tmp_path, monkeypatch, data)
    assert victim.read_text() == "existing"


def test_model_download_verifies_digest_before_extracting(tmp_path, monkeypatch):
    data = archive_bytes("DatasetSynthetic/checkpoint.pth")
    with pytest.raises(ValueError, match="SHA-256"):
        download(tmp_path, monkeypatch, data, "0" * 64)
    assert not (tmp_path / "DatasetSynthetic/checkpoint.pth").exists()


def test_model_download_accepts_verified_archive(tmp_path, monkeypatch):
    data = archive_bytes("DatasetSynthetic/checkpoint.pth")
    download(tmp_path, monkeypatch, data)
    assert (tmp_path / "DatasetSynthetic/checkpoint.pth").read_bytes() == b"synthetic"


@pytest.mark.parametrize("failure", ["download", "digest", "install"])
def test_failed_model_upgrade_preserves_legacy_cache(tmp_path, monkeypatch, failure):
    cached = tmp_path / "DatasetSynthetic"
    cached.mkdir()
    (cached / "checkpoint.pth").write_bytes(b"previous")
    (cached / "model_version.json").write_text('{"url": "https://example.invalid/synthetic.zip"}')
    data = archive_bytes("DatasetSynthetic/checkpoint.pth", b"replacement")
    if failure == "install":
        real_replace = models.os.replace
        def fail_install(source, destination):
            if models.Path(source).parent.name == "extracted":
                raise OSError("synthetic install failure")
            return real_replace(source, destination)
        monkeypatch.setattr(models.os, "replace", fail_install)
    with pytest.raises((models.requests.ConnectionError, ValueError, OSError)):
        download(tmp_path, monkeypatch, data,
                 digest="0" * 64 if failure == "digest" else None,
                 fail_download=failure == "download")
    assert (cached / "checkpoint.pth").read_bytes() == b"previous"
    assert (cached / "model_version.json").read_text() == '{"url": "https://example.invalid/synthetic.zip"}'


def test_verified_upgrade_replaces_legacy_cache(tmp_path, monkeypatch):
    cached = tmp_path / "DatasetSynthetic"
    cached.mkdir()
    (cached / "checkpoint.pth").write_bytes(b"previous")
    data = archive_bytes("DatasetSynthetic/checkpoint.pth", b"replacement")
    download(tmp_path, monkeypatch, data)
    assert (cached / "checkpoint.pth").read_bytes() == b"replacement"


def test_verified_model_archive_ignores_apple_metadata(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('DatasetSynthetic/checkpoint.pth', b'synthetic')
        archive.writestr('__MACOSX/._DatasetSynthetic', b'metadata')
        archive.writestr('__MACOSX/../../outside.txt', b'metadata must never be extracted')
        archive.writestr('.DS_Store', b'metadata')
    download(tmp_path, monkeypatch, buffer.getvalue())
    assert (tmp_path / 'DatasetSynthetic/checkpoint.pth').read_bytes() == b'synthetic'
    assert not (tmp_path / '__MACOSX').exists()
    assert not (tmp_path / '.DS_Store').exists()
    assert not (tmp_path / 'outside.txt').exists()
