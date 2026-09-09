import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_idc_dicom_seg.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_idc_dicom_seg", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cohort_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    selection_csv = project / "selection.csv"
    qc_csv = project / "segmentation_validation.csv"
    patients = ["Lung_Dx-A0001", "Lung_Dx-A0002", "Lung_Dx-A0003"]
    _write_csv(
        selection_csv,
        ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
        [
            {
                "PatientID": patient,
                "StudyInstanceUID": f"1.2.3.{index}",
                "SeriesInstanceUID": f"1.2.840.{index}",
            }
            for index, patient in enumerate(patients, start=1)
        ],
    )
    _write_csv(
        qc_csv,
        [
            "PatientID",
            "status",
            "native_path",
            "suv4_path",
            "native_voxels",
            "suv4_voxels",
            "error",
        ],
        [
            {
                "PatientID": patient,
                "status": "ok",
                "native_path": project / "results" / "native" / f"{patient}_native.nii.gz",
                "suv4_path": project / "results" / "suv4" / f"{patient}_suv4.nii.gz",
                "native_voxels": 10 + index,
                "suv4_voxels": 0 if index == 3 else index,
                "error": "",
            }
            for index, patient in enumerate(patients, start=1)
        ],
    )
    return project, selection_csv, qc_csv


def test_prepare_records_empty_suv4_without_conversion(tmp_path):
    module = _load_script()
    project, selection_csv, qc_csv = _cohort_inputs(tmp_path)

    result = module.prepare_export(
        selection_csv=selection_csv,
        qc_csv=qc_csv,
        project_dir=project,
        output_root=project / "results" / "dicom-seg",
        source_commit="39a2139",
        exporter_commit="abcdef0123456789",
        dcmqi_sha256="1" * 64,
        dicom3tools_sha256="2" * 64,
    )

    assert result["expected_native"] == 3
    assert result["expected_suv4"] == 2
    assert result["empty_suv4"] == ["Lung_Dx-A0003"]
    tasks = [json.loads(line) for line in Path(result["tasks_file"]).read_text().splitlines()]
    assert len(tasks) == 5
    assert all(task["exporter_commit"] == "abcdef0123456789" for task in tasks)
    assert all(
        task["tool_sha256"] == {"dcmqi": "1" * 64, "dicom3tools": "2" * 64}
        for task in tasks
    )
    assert {(task["patient_id"], task["variant"]) for task in tasks} == {
        ("Lung_Dx-A0001", "native"),
        ("Lung_Dx-A0001", "suv4"),
        ("Lung_Dx-A0002", "native"),
        ("Lung_Dx-A0002", "suv4"),
        ("Lung_Dx-A0003", "native"),
    }
    empty_report = json.loads(
        (project / "results" / "dicom-seg" / "qc" / "reports" / "Lung_Dx-A0003_suv4.json").read_text()
    )
    assert empty_report["status"] == "NO_SEGMENT_ABOVE_THRESHOLD"
    assert empty_report["exporter_commit"] == "abcdef0123456789"


def _write_report(
    report_dir: Path,
    *,
    patient: str,
    variant: str,
    sop_uid: str,
    series_uid: str,
    dciodvfy_errors: list[str] | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient_id": patient,
        "variant": variant,
        "status": "ok",
        "sop_instance_uid": sop_uid,
        "series_instance_uid": series_uid,
        "source_sop_instances": 2,
        "referenced_sop_instances": 2,
        "geometry": {"input_voxels": 5, "output_voxels": 5},
        "roundtrip": {"equal": True, "dice": 1.0, "differing_voxels": 0},
        "dciodvfy": {"returncode": 0, "errors": dciodvfy_errors or []},
        "dcentvfy": {"returncode": 0, "errors": []},
        "subset_ok": True,
    }
    (report_dir / f"{patient}_{variant}.json").write_text(json.dumps(payload))


def test_summarize_accepts_unique_exactly_validated_outputs(tmp_path):
    module = _load_script()
    report_dir = tmp_path / "qc" / "reports"
    _write_report(
        report_dir,
        patient="Lung_Dx-A0001",
        variant="native",
        sop_uid="1.2.3.1",
        series_uid="1.2.4.1",
    )
    _write_report(
        report_dir,
        patient="Lung_Dx-A0001",
        variant="suv4",
        sop_uid="1.2.3.2",
        series_uid="1.2.4.2",
    )

    summary = module.summarize_reports(
        report_dir=report_dir,
        expected_native=1,
        expected_suv4=1,
        expected_empty_suv4=[],
    )

    assert summary["status"] == "ok"
    assert summary["dicom_seg_objects"] == 2
    assert summary["unique_sop_instance_uids"] == 2
    assert summary["unique_series_instance_uids"] == 2


def test_summarize_rejects_any_failed_validator(tmp_path):
    module = _load_script()
    report_dir = tmp_path / "qc" / "reports"
    _write_report(
        report_dir,
        patient="Lung_Dx-A0001",
        variant="native",
        sop_uid="1.2.3.1",
        series_uid="1.2.4.1",
        dciodvfy_errors=["Error - bad reference"],
    )

    with pytest.raises(ValueError, match="dciodvfy"):
        module.summarize_reports(
            report_dir=report_dir,
            expected_native=1,
            expected_suv4=0,
            expected_empty_suv4=[],
        )


def test_summarize_requires_exact_empty_suv4_records(tmp_path):
    module = _load_script()
    report_dir = tmp_path / "qc" / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "Lung_Dx-A0003_suv4.json").write_text(
        json.dumps(
            {
                "patient_id": "Lung_Dx-A0003",
                "variant": "suv4",
                "status": "NO_SEGMENT_ABOVE_THRESHOLD",
                "source_voxels": 0,
            }
        )
    )

    summary = module.summarize_reports(
        report_dir=report_dir,
        expected_native=0,
        expected_suv4=0,
        expected_empty_suv4=["Lung_Dx-A0003"],
    )

    assert summary["empty_suv4"] == ["Lung_Dx-A0003"]


def test_summarize_rejects_missing_source_instance_references(tmp_path):
    module = _load_script()
    report_dir = tmp_path / "qc" / "reports"
    _write_report(
        report_dir,
        patient="Lung_Dx-A0001",
        variant="native",
        sop_uid="1.2.3.1",
        series_uid="1.2.4.1",
    )
    path = report_dir / "Lung_Dx-A0001_native.json"
    report = json.loads(path.read_text())
    report["referenced_sop_instances"] = 1
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="source references"):
        module.summarize_reports(
            report_dir=report_dir,
            expected_native=1,
            expected_suv4=0,
            expected_empty_suv4=[],
        )


def test_summarize_rejects_unknown_report_status(tmp_path):
    module = _load_script()
    report_dir = tmp_path / "qc" / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "Lung_Dx-A0001_native.json").write_text(
        json.dumps(
            {
                "patient_id": "Lung_Dx-A0001",
                "variant": "native",
                "status": "skipped",
            }
        )
    )

    with pytest.raises(ValueError, match="Unexpected report status"):
        module.summarize_reports(
            report_dir=report_dir,
            expected_native=0,
            expected_suv4=0,
            expected_empty_suv4=[],
        )


def _write_roundtrip_image(path: Path, array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((-5.0, 7.0, 11.0))
    sitk.WriteImage(image, str(path))
    return image


def test_exact_roundtrip_rejects_changed_positive_label_value(tmp_path):
    module = _load_script()
    reference_array = np.zeros((2, 3, 4), dtype=np.uint8)
    reference_array[0, 1, 2] = 1
    changed_array = reference_array.copy()
    changed_array[0, 1, 2] = 2
    reference_path = tmp_path / "reference.nrrd"
    changed_path = tmp_path / "changed.nrrd"
    _write_roundtrip_image(reference_path, reference_array)
    _write_roundtrip_image(changed_path, changed_array)

    report = module._exact_roundtrip(reference_path, changed_path)

    assert report["equal"] is False
    assert report["binary_values"] is False


def test_exact_roundtrip_rejects_mirrored_direction_with_same_corners(tmp_path):
    module = _load_script()
    array = np.zeros((2, 3, 4), dtype=np.uint8)
    array[0, 1, 2] = 1
    reference_path = tmp_path / "reference.nrrd"
    mirrored_path = tmp_path / "mirrored.nrrd"
    reference = _write_roundtrip_image(reference_path, array)
    mirrored = sitk.GetImageFromArray(array)
    mirrored.SetSpacing(reference.GetSpacing())
    mirrored.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    mirrored.SetOrigin(
        (
            reference.GetOrigin()[0]
            + (reference.GetSize()[0] - 1) * reference.GetSpacing()[0],
            reference.GetOrigin()[1],
            reference.GetOrigin()[2],
        )
    )
    sitk.WriteImage(mirrored, str(mirrored_path))

    report = module._exact_roundtrip(reference_path, mirrored_path)

    assert report["max_corner_distance_mm"] == pytest.approx(0.0)
    assert report["geometry_equal"] is False
    assert report["equal"] is False


def test_verify_output_artifacts_rejects_replaced_dicom(tmp_path):
    module = _load_script()
    output_root = tmp_path / "dicom-seg"
    report_dir = output_root / "qc" / "reports"
    output_path = output_root / "native" / "Lung_Dx-A0001_LION_FDG_native.dcm"
    output_path.parent.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    output_path.write_bytes(b"validated-seg")
    report = {
        "patient_id": "Lung_Dx-A0001",
        "variant": "native",
        "status": "ok",
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "exporter_commit": "abcdef0123456789",
        "source_commit": "39a2139",
        "tool_sha256": {"dcmqi": "1" * 64, "dicom3tools": "2" * 64},
    }
    (report_dir / "Lung_Dx-A0001_native.json").write_text(json.dumps(report))

    verified = module.verify_output_artifacts(
        report_dir=report_dir,
        output_root=output_root,
        exporter_commit="abcdef0123456789",
        tool_sha256={"dcmqi": "1" * 64, "dicom3tools": "2" * 64},
    )
    assert verified["objects"] == 1

    output_path.write_bytes(b"tamperedd-seg")
    with pytest.raises(ValueError, match="SHA-256"):
        module.verify_output_artifacts(
            report_dir=report_dir,
            output_root=output_root,
            exporter_commit="abcdef0123456789",
            tool_sha256={"dcmqi": "1" * 64, "dicom3tools": "2" * 64},
        )


def test_dicom3tools_command_uses_container_absolute_binary_path():
    module = _load_script()

    command = module._dciodvfy_command(
        "singularity",
        "/tools/dicom3tools.sif",
        "/output/seg.dcm",
    )

    assert command == [
        "singularity",
        "exec",
        "--bind",
        "/data2:/data2",
        "/tools/dicom3tools.sif",
        "/usr/src/dicom3tools/bin/1.4.4.0.x8664/dciodvfy",
        "/output/seg.dcm",
    ]


def test_dcmqi_conversion_keeps_empty_source_frames_for_exact_roundtrip():
    module = _load_script()

    command = module._dcmqi_conversion_command(
        "singularity",
        "/tools/dcmqi.sif",
        source_dir="/source",
        canonical_path="/work/mask.nrrd",
        metadata_path="/work/metadata.json",
        output_path="/output/seg.dcm",
    )

    assert command[-2:] == ["--skip", "0"]
    assert "--skipEmptySlices" not in command
    assert command[2:4] == ["--bind", "/data2:/data2"]
