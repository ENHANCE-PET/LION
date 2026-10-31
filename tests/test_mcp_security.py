"""Use synthetic text files to exercise organization plan boundaries."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lion_mcp_security_server",
    Path(__file__).resolve().parents[1] / "lion-mcp/src/lion_mcp/server.py",
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def organize(source_root, target_root, source, subject="subject", name="PT_scan.nii", copy=True):
    plan = {"subjects": [{"subject_id": subject, "files": [{"source": str(source), "target_name": name, "modality": "PT"}]}]}
    return json.loads(server.organize_for_lion(str(source_root), str(target_root), json.dumps(plan), copy_files=copy))


def test_organize_copies_an_authorized_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "scan.nii"
    image.write_text("synthetic")
    target = tmp_path / "target"
    assert organize(source, target, image)["success"]
    assert (target / "subject/PT_scan.nii").read_text() == "synthetic"


@pytest.mark.parametrize("copy", [True, False])
def test_organize_rejects_sources_outside_source_root(tmp_path, copy):
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret.nii"
    secret.write_text("private")
    target = tmp_path / "target"
    assert not organize(source, target, secret, copy=copy)["success"]
    assert secret.read_text() == "private"
    assert not (target / "subject/PT_scan.nii").exists()


@pytest.mark.parametrize("subject,name", [("../escape", "PT_scan.nii"), ("subject", "../../PT_escape.nii"), ("subject", "PT_../../escape.nii"), ("subject", r"PT_..\escape.nii"), (".. ", "PT_scan.nii"), ("/absolute", "PT_scan.nii")])
def test_organize_rejects_path_components(tmp_path, subject, name):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "scan.nii"
    image.write_text("synthetic")
    assert not organize(source, tmp_path / "target", image, subject, name)["success"]


def test_organize_rejects_source_symlink_escape(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret.nii"
    secret.write_text("private")
    link = source / "link.nii"
    link.symlink_to(secret)
    assert not organize(source, tmp_path / "target", link)["success"]


def test_organize_does_not_follow_destination_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "scan.nii"
    image.write_text("synthetic")
    target = tmp_path / "target"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (target / "subject").symlink_to(outside, target_is_directory=True)
    assert not organize(source, target, image)["success"]
    assert not (outside / "PT_scan.nii").exists()


def test_organize_does_not_overwrite_existing_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = source / "scan.nii"
    image.write_text("synthetic")
    target = tmp_path / "target"
    (target / "subject").mkdir(parents=True)
    existing = target / "subject/PT_scan.nii"
    existing.write_text("existing")
    assert not organize(source, target, image)["success"]
    assert existing.read_text() == "existing"
