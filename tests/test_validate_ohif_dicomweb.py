import hashlib
import importlib.util
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_ohif_dicomweb.py"
SPEC = importlib.util.spec_from_file_location("validate_ohif_dicomweb", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


STUDY_UID = generate_uid()
SOURCE_SERIES_UID = generate_uid()
NATIVE_SERIES_UID = generate_uid()
SUV4_SERIES_UID = generate_uid()
SOURCE_SOPS = [generate_uid(), generate_uid()]
NATIVE_SOP = generate_uid()
SUV4_SOP = generate_uid()
OHIF_DIGEST = validator.EXPECTED_OHIF_REFERENCE.rsplit("@", 1)[1]
ORTHANC_DIGEST = validator.EXPECTED_ORTHANC_REFERENCE.rsplit("@", 1)[1]


def _write_dicom(
    path: Path,
    *,
    series_uid: str,
    sop_uid: str,
    modality: str,
    description: str,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.StudyInstanceUID = STUDY_UID
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPInstanceUID = sop_uid
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.PatientID = "Lung_Dx-A0164"
    dataset.Modality = modality
    dataset.SeriesDescription = description
    dataset.save_as(path, enforce_file_format=True)


def _tag(value):
    return {"Value": [value]}


def _container(image: str, digest: str) -> dict[str, object]:
    repository = image.split(":", 1)[0]
    return {
        "configured_reference": f"{image}@{digest}",
        "image_id": "sha256:" + "c" * 64,
        "repo_digests": [f"{repository}@{digest}"],
        "running": True,
    }


@pytest.fixture
def smoke_files(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    source_dir = tmp_path / "source"
    seg_dir = tmp_path / "seg"
    source_dir.mkdir()
    seg_dir.mkdir()
    for index, sop_uid in enumerate(SOURCE_SOPS):
        _write_dicom(
            source_dir / f"source-{index}.dcm",
            series_uid=SOURCE_SERIES_UID,
            sop_uid=sop_uid,
            modality="PT",
            description="PET",
        )
    paths = {
        "native": seg_dir / "native.dcm",
        "suv4": seg_dir / "suv4.dcm",
    }
    _write_dicom(
        paths["native"],
        series_uid=NATIVE_SERIES_UID,
        sop_uid=NATIVE_SOP,
        modality="SEG",
        description="LION FDG native DICOM SEG",
    )
    _write_dicom(
        paths["suv4"],
        series_uid=SUV4_SERIES_UID,
        sop_uid=SUV4_SOP,
        modality="SEG",
        description="LION FDG SUV>=4 DICOM SEG",
    )
    return source_dir, seg_dir, paths


def _qido_get(url: str, *, wrong_source_sop: bool = False):
    if url.endswith("/studies"):
        return [{"0020000D": _tag(STUDY_UID)}]
    if url.endswith(f"/studies/{STUDY_UID}/series"):
        return [
            {
                "00080060": _tag("PT"),
                "0008103E": _tag("PET"),
                "00201209": _tag(2),
                "0020000E": _tag(SOURCE_SERIES_UID),
            },
            {
                "00080060": _tag("SEG"),
                "0008103E": _tag("LION FDG native DICOM SEG"),
                "00201209": _tag(1),
                "0020000E": _tag(NATIVE_SERIES_UID),
            },
            {
                "00080060": _tag("SEG"),
                "0008103E": _tag("LION FDG SUV>=4 DICOM SEG"),
                "00201209": _tag(1),
                "0020000E": _tag(SUV4_SERIES_UID),
            },
        ]
    series_uid = url.rsplit("/", 2)[-2]
    expected = {
        SOURCE_SERIES_UID: SOURCE_SOPS,
        NATIVE_SERIES_UID: [NATIVE_SOP],
        SUV4_SERIES_UID: [SUV4_SOP],
    }[series_uid]
    if wrong_source_sop and series_uid == SOURCE_SERIES_UID:
        expected = [SOURCE_SOPS[0], generate_uid()]
    return [{"00080018": _tag(uid)} for uid in expected]


def _run(monkeypatch, smoke_files, *, wrong_source_sop=False, wrong_wado=False):
    source_dir, seg_dir, paths = smoke_files
    monkeypatch.setattr(
        validator,
        "_json_get",
        lambda url: _qido_get(url, wrong_source_sop=wrong_source_sop),
    )

    def dicom_get(url: str) -> bytes:
        path = paths["native"] if NATIVE_SOP in url else paths["suv4"]
        payload = path.read_bytes()
        return payload + b"wrong" if wrong_wado and NATIVE_SOP in url else payload

    monkeypatch.setattr(validator, "_dicom_get", dicom_get)
    monkeypatch.setattr(
        validator,
        "_inspect_container",
        lambda name: _container(
            "ohif/app:v3.12.15"
            if name.endswith("-ohif")
            else "orthancteam/orthanc:26.8.2",
            OHIF_DIGEST if name.endswith("-ohif") else ORTHANC_DIGEST,
        ),
    )
    return validator.validate(
        source_dir=source_dir,
        seg_dir=seg_dir,
        dicomweb_url="http://127.0.0.1:3001/dicom-web",
        exporter_commit="d" * 40,
        ohif_container="lionz-ohif-smoke-ohif",
        orthanc_container="lionz-ohif-smoke-orthanc",
    )


def test_rejects_wrong_qido_sop_uid_even_when_series_counts_match(
    monkeypatch, smoke_files
):
    with pytest.raises(ValueError, match="SOP Instance UID inventory differs"):
        _run(monkeypatch, smoke_files, wrong_source_sop=True)


def test_rejects_wado_seg_bytes_that_do_not_match_local_object(
    monkeypatch, smoke_files
):
    with pytest.raises(ValueError, match="WADO bytes differ"):
        _run(monkeypatch, smoke_files, wrong_wado=True)


def test_records_exact_qido_wado_and_observed_container_evidence(
    monkeypatch, smoke_files
):
    report = _run(monkeypatch, smoke_files)

    assert report["source_sop_instance_uids"] == sorted(SOURCE_SOPS)
    objects = {item["variant"]: item for item in report["seg_objects"]}
    assert objects["native"]["wado_sha256"] == hashlib.sha256(
        smoke_files[2]["native"].read_bytes()
    ).hexdigest()
    assert report["ohif"]["configured_reference"].endswith(f"@{OHIF_DIGEST}")
    assert report["orthanc"]["configured_reference"].endswith(
        f"@{ORTHANC_DIGEST}"
    )
