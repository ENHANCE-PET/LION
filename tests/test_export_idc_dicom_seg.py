import csv
import importlib.util
import json
from pathlib import Path

import pytest


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
    )

    assert result["expected_native"] == 3
    assert result["expected_suv4"] == 2
    assert result["empty_suv4"] == ["Lung_Dx-A0003"]
    tasks = [json.loads(line) for line in Path(result["tasks_file"]).read_text().splitlines()]
    assert len(tasks) == 5
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
