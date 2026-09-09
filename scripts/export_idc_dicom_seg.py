#!/usr/bin/env python3
"""Export LION LUNG-PET-CT-Dx masks as validated DICOM SEG objects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk

from lionz.dicom_seg import (
    build_dcmqi_metadata,
    canonicalize_binary_mask,
    load_source_series,
    parse_dcentvfy,
    parse_dciodvfy,
    physical_corner_distance,
    uid_from_key,
    validate_seg_dataset,
)


EMPTY_STATUS = "NO_SEGMENT_ABOVE_THRESHOLD"


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _write_json_lines(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def _integer(row: dict[str, str], field: str, patient_id: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} for {patient_id}") from exc
    if value < 0:
        raise ValueError(f"Negative {field} for {patient_id}")
    return value


def prepare_export(
    *,
    selection_csv: str | Path,
    qc_csv: str | Path,
    project_dir: str | Path,
    output_root: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    """Build the deterministic cohort task manifest and empty-mask records."""

    project = Path(project_dir)
    output = Path(output_root)
    selection_rows = _read_csv(selection_csv)
    qc_rows = _read_csv(qc_csv)
    selection_by_patient = {row["PatientID"]: row for row in selection_rows}
    qc_by_patient = {row["PatientID"]: row for row in qc_rows}
    if len(selection_by_patient) != len(selection_rows):
        raise ValueError("Selection manifest contains duplicate PatientID values")
    if len(qc_by_patient) != len(qc_rows):
        raise ValueError("Segmentation QC contains duplicate PatientID values")
    if selection_by_patient.keys() != qc_by_patient.keys():
        missing_qc = sorted(selection_by_patient.keys() - qc_by_patient.keys())
        missing_selection = sorted(qc_by_patient.keys() - selection_by_patient.keys())
        raise ValueError(
            "Selection/QC PatientID mismatch: "
            f"missing QC={missing_qc}, missing selection={missing_selection}"
        )
    if not source_commit.strip():
        raise ValueError("source_commit must not be empty")

    report_dir = output / "qc" / "reports"
    tasks: list[dict[str, Any]] = []
    empty_suv4: list[str] = []
    for patient_id in sorted(selection_by_patient):
        selection = selection_by_patient[patient_id]
        qc = qc_by_patient[patient_id]
        if qc.get("status") != "ok" or qc.get("error", "").strip():
            raise ValueError(f"Segmentation QC is not clean for {patient_id}")
        native_voxels = _integer(qc, "native_voxels", patient_id)
        suv4_voxels = _integer(qc, "suv4_voxels", patient_id)
        if native_voxels == 0:
            raise ValueError(f"Native mask is unexpectedly empty for {patient_id}")

        common = {
            "patient_id": patient_id,
            "expected_study_uid": selection["StudyInstanceUID"],
            "expected_series_uid": selection["SeriesInstanceUID"],
            "source_dir": str(project / "raw" / patient_id / "PT"),
            "source_commit": source_commit,
            "native_mask_path": qc["native_path"],
        }
        for variant, voxels in (("native", native_voxels), ("suv4", suv4_voxels)):
            report_path = report_dir / f"{patient_id}_{variant}.json"
            if variant == "suv4" and voxels == 0:
                empty_suv4.append(patient_id)
                _atomic_json(
                    report_path,
                    {
                        "patient_id": patient_id,
                        "variant": variant,
                        "status": EMPTY_STATUS,
                        "source_voxels": 0,
                        "mask_path": qc["suv4_path"],
                    },
                )
                continue
            mask_path = qc["native_path"] if variant == "native" else qc["suv4_path"]
            tasks.append(
                {
                    **common,
                    "variant": variant,
                    "mask_path": mask_path,
                    "expected_voxels": voxels,
                    "output_path": str(output / variant / f"{patient_id}_LION_FDG_{variant}.dcm"),
                    "canonical_path": str(output / "work" / "canonical" / f"{patient_id}_{variant}.nrrd"),
                    "metadata_path": str(output / "work" / "metadata" / f"{patient_id}_{variant}.json"),
                    "roundtrip_dir": str(output / "work" / "roundtrip" / f"{patient_id}_{variant}"),
                    "report_path": str(report_path),
                    "validator_log_dir": str(output / "qc" / "validators"),
                }
            )

    tasks_path = output / "qc" / "dicom_seg_tasks.jsonl"
    _write_json_lines(tasks_path, tasks)
    result = {
        "status": "prepared",
        "source_commit": source_commit,
        "patients": len(selection_by_patient),
        "expected_native": sum(task["variant"] == "native" for task in tasks),
        "expected_suv4": sum(task["variant"] == "suv4" for task in tasks),
        "empty_suv4": empty_suv4,
        "tasks": len(tasks),
        "tasks_file": str(tasks_path),
        "report_dir": str(report_dir),
    }
    _atomic_json(output / "qc" / "dicom_seg_expected.json", result)
    return result


def _run_command(arguments: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout)
    return completed


def _singularity_command(
    singularity: str,
    sif: str | Path,
    executable: str,
    *arguments: str | Path,
) -> list[str]:
    return [singularity, "exec", str(sif), executable, *(str(item) for item in arguments)]


def _exact_roundtrip(reference_path: Path, roundtrip_path: Path) -> dict[str, Any]:
    reference = sitk.ReadImage(str(reference_path))
    roundtrip = sitk.ReadImage(str(roundtrip_path))
    reference_array = sitk.GetArrayFromImage(reference) > 0
    roundtrip_array = sitk.GetArrayFromImage(roundtrip) > 0
    same_shape = reference_array.shape == roundtrip_array.shape
    differing = (
        int(np.count_nonzero(reference_array != roundtrip_array))
        if same_shape
        else -1
    )
    intersection = (
        int(np.count_nonzero(reference_array & roundtrip_array))
        if same_shape
        else 0
    )
    denominator = int(np.count_nonzero(reference_array)) + int(
        np.count_nonzero(roundtrip_array)
    )
    dice = 1.0 if denominator == 0 else (2.0 * intersection / denominator)
    geometry_distance = (
        physical_corner_distance(reference, roundtrip)
        if reference.GetDimension() == roundtrip.GetDimension()
        else float("inf")
    )
    equal = (
        same_shape
        and differing == 0
        and geometry_distance <= 0.05
        and reference.GetSize() == roundtrip.GetSize()
    )
    return {
        "equal": equal,
        "dice": dice,
        "differing_voxels": differing,
        "reference_voxels": int(np.count_nonzero(reference_array)),
        "roundtrip_voxels": int(np.count_nonzero(roundtrip_array)),
        "max_corner_distance_mm": geometry_distance,
    }


def _select_task(
    tasks_file: str | Path,
    *,
    task_index: int | None = None,
    patient_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    tasks = [json.loads(line) for line in Path(tasks_file).read_text().splitlines() if line.strip()]
    if task_index is not None:
        if task_index < 1 or task_index > len(tasks):
            raise ValueError(f"task_index must be between 1 and {len(tasks)}")
        return tasks[task_index - 1]
    matches = [
        task
        for task in tasks
        if task["patient_id"] == patient_id and task["variant"] == variant
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one task for {patient_id}/{variant}, found {len(matches)}")
    return matches[0]


def convert_task(
    task: dict[str, Any],
    *,
    dcmqi_sif: str | Path,
    dicom3tools_sif: str | Path,
    singularity: str = "singularity",
) -> dict[str, Any]:
    """Convert and independently validate exactly one cohort task."""

    patient_id = task["patient_id"]
    variant = task["variant"]
    report_path = Path(task["report_path"])
    report: dict[str, Any] = {
        "patient_id": patient_id,
        "variant": variant,
        "status": "error",
    }
    try:
        source = load_source_series(task["source_dir"], task["expected_series_uid"])
        if source.patient_id != patient_id:
            raise ValueError(f"Source PatientID {source.patient_id} != {patient_id}")
        if source.study_instance_uid != task["expected_study_uid"]:
            raise ValueError("Source StudyInstanceUID differs from cohort manifest")

        mask_path = Path(task["mask_path"])
        mask = sitk.ReadImage(str(mask_path))
        canonical, geometry = canonicalize_binary_mask(mask, source.image)
        if geometry.input_voxels != int(task["expected_voxels"]):
            raise ValueError(
                f"Mask voxel count {geometry.input_voxels} != QC {task['expected_voxels']}"
            )
        if geometry.input_voxels == 0:
            raise ValueError("Conversion task cannot contain an empty mask")
        canonical_path = Path(task["canonical_path"])
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(canonical, str(canonical_path), True)

        metadata = build_dcmqi_metadata(
            variant,
            task["source_commit"],
            uid_from_key(f"{patient_id}:{variant}"),
        )
        metadata_path = Path(task["metadata_path"])
        _atomic_json(metadata_path, metadata)

        output_path = Path(task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.stem}.tmp.dcm")
        if temporary_output.exists():
            temporary_output.unlink()
        log_dir = Path(task["validator_log_dir"])
        dcmqi_log = log_dir / f"{patient_id}_{variant}_itkimage2segimage.log"
        conversion = _run_command(
            _singularity_command(
                singularity,
                dcmqi_sif,
                "itkimage2segimage",
                "--inputDICOMDirectory",
                source.directory,
                "--inputImageList",
                canonical_path,
                "--inputMetadata",
                metadata_path,
                "--outputDICOM",
                temporary_output,
                "--segmentationType",
                "binary",
                "--compress",
                "none",
                "--skipEmptySlices",
                "0",
            ),
            dcmqi_log,
        )
        if conversion.returncode != 0 or not temporary_output.is_file():
            raise RuntimeError(
                f"itkimage2segimage failed with return code {conversion.returncode}"
            )

        seg = validate_seg_dataset(temporary_output, source, variant)
        dciodvfy_log = log_dir / f"{patient_id}_{variant}_dciodvfy.log"
        dciodvfy = _run_command(
            _singularity_command(
                singularity, dicom3tools_sif, "dciodvfy", "-new", temporary_output
            ),
            dciodvfy_log,
        )
        dciodvfy_errors = list(parse_dciodvfy(dciodvfy.stdout))
        if dciodvfy.returncode != 0 or dciodvfy_errors:
            raise RuntimeError(
                "dciodvfy failed: "
                f"returncode={dciodvfy.returncode}, errors={dciodvfy_errors}"
            )

        dcentvfy_log = log_dir / f"{patient_id}_{variant}_dcentvfy.log"
        dcentvfy = _run_command(
            _singularity_command(
                singularity,
                dicom3tools_sif,
                "dcentvfy",
                temporary_output,
                *source.files,
            ),
            dcentvfy_log,
        )
        dcentvfy_errors = list(parse_dcentvfy(dcentvfy.stdout))
        if dcentvfy.returncode != 0 or dcentvfy_errors:
            raise RuntimeError(
                "dcentvfy failed: "
                f"returncode={dcentvfy.returncode}, errors={dcentvfy_errors}"
            )

        roundtrip_dir = Path(task["roundtrip_dir"])
        roundtrip_dir.mkdir(parents=True, exist_ok=True)
        for stale in roundtrip_dir.glob("roundtrip*.nrrd"):
            stale.unlink()
        roundtrip_log = log_dir / f"{patient_id}_{variant}_segimage2itkimage.log"
        roundtrip_command = _run_command(
            _singularity_command(
                singularity,
                dcmqi_sif,
                "segimage2itkimage",
                "--inputDICOM",
                temporary_output,
                "--outputDirectory",
                roundtrip_dir,
                "--outputType",
                "nrrd",
                "--prefix",
                "roundtrip",
            ),
            roundtrip_log,
        )
        roundtrip_files = sorted(roundtrip_dir.glob("roundtrip*.nrrd"))
        if roundtrip_command.returncode != 0 or len(roundtrip_files) != 1:
            raise RuntimeError(
                "segimage2itkimage failed or produced an unexpected number of masks: "
                f"returncode={roundtrip_command.returncode}, files={len(roundtrip_files)}"
            )
        roundtrip = _exact_roundtrip(canonical_path, roundtrip_files[0])
        if not roundtrip["equal"] or roundtrip["dice"] != 1.0:
            raise ValueError(f"DICOM SEG round trip differs from canonical mask: {roundtrip}")

        subset_ok = True
        if variant == "suv4":
            native = sitk.ReadImage(str(task["native_mask_path"]))
            canonical_native, _ = canonicalize_binary_mask(native, source.image)
            suv_array = sitk.GetArrayFromImage(canonical) > 0
            native_array = sitk.GetArrayFromImage(canonical_native) > 0
            subset_ok = not bool(np.any(suv_array & ~native_array))
            if not subset_ok:
                raise ValueError("SUV>=4 mask is not a subset of the native mask")

        temporary_output.replace(output_path)
        report.update(
            {
                "status": "ok",
                "output_path": str(output_path),
                "source_series_instance_uid": source.series_instance_uid,
                "source_sop_instances": len(source.sop_instance_uids),
                "referenced_sop_instances": len(seg.referenced_sop_instance_uids),
                "sop_instance_uid": seg.sop_instance_uid,
                "series_instance_uid": seg.series_instance_uid,
                "frames": seg.frame_count,
                "geometry": {
                    "input_voxels": geometry.input_voxels,
                    "output_voxels": geometry.output_voxels,
                    "max_corner_distance_mm": geometry.max_corner_distance_mm,
                },
                "dciodvfy": {
                    "returncode": dciodvfy.returncode,
                    "errors": dciodvfy_errors,
                    "log": str(dciodvfy_log),
                },
                "dcentvfy": {
                    "returncode": dcentvfy.returncode,
                    "errors": dcentvfy_errors,
                    "log": str(dcentvfy_log),
                },
                "roundtrip": roundtrip,
                "subset_ok": subset_ok,
            }
        )
        _atomic_json(report_path, report)
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(report_path, report)
        raise


def validate_existing_task(
    task: dict[str, Any],
    *,
    dicom3tools_sif: str | Path,
    singularity: str = "singularity",
) -> dict[str, Any]:
    """Re-run structural and dicom3tools validation on an existing object."""

    source = load_source_series(task["source_dir"], task["expected_series_uid"])
    seg = validate_seg_dataset(task["output_path"], source, task["variant"])
    log_dir = Path(task["validator_log_dir"])
    prefix = f"{task['patient_id']}_{task['variant']}_revalidate"
    dciodvfy = _run_command(
        _singularity_command(
            singularity, dicom3tools_sif, "dciodvfy", "-new", task["output_path"]
        ),
        log_dir / f"{prefix}_dciodvfy.log",
    )
    dcentvfy = _run_command(
        _singularity_command(
            singularity,
            dicom3tools_sif,
            "dcentvfy",
            task["output_path"],
            *source.files,
        ),
        log_dir / f"{prefix}_dcentvfy.log",
    )
    result = {
        "sop_instance_uid": seg.sop_instance_uid,
        "dciodvfy_returncode": dciodvfy.returncode,
        "dciodvfy_errors": list(parse_dciodvfy(dciodvfy.stdout)),
        "dcentvfy_returncode": dcentvfy.returncode,
        "dcentvfy_errors": list(parse_dcentvfy(dcentvfy.stdout)),
    }
    if any(
        (
            result["dciodvfy_returncode"],
            result["dciodvfy_errors"],
            result["dcentvfy_returncode"],
            result["dcentvfy_errors"],
        )
    ):
        raise ValueError(f"Existing DICOM SEG failed revalidation: {result}")
    return result


def summarize_reports(
    *,
    report_dir: str | Path,
    expected_native: int,
    expected_suv4: int,
    expected_empty_suv4: list[str],
) -> dict[str, Any]:
    """Apply cohort-wide count, uniqueness, and validation gates."""

    reports = [json.loads(path.read_text()) for path in sorted(Path(report_dir).glob("*.json"))]
    identities = [(report.get("patient_id"), report.get("variant")) for report in reports]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate patient/variant reports found")
    failed = [report for report in reports if report.get("status") == "error"]
    if failed:
        raise ValueError(f"Conversion failures found: {failed}")
    empty = sorted(
        report["patient_id"]
        for report in reports
        if report.get("status") == EMPTY_STATUS and report.get("variant") == "suv4"
    )
    if empty != sorted(expected_empty_suv4):
        raise ValueError(f"Unexpected empty SUV>=4 records: {empty}")

    successful = [report for report in reports if report.get("status") == "ok"]
    native = [report for report in successful if report.get("variant") == "native"]
    suv4 = [report for report in successful if report.get("variant") == "suv4"]
    if len(native) != expected_native or len(suv4) != expected_suv4:
        raise ValueError(
            "DICOM SEG count mismatch: "
            f"native={len(native)}/{expected_native}, suv4={len(suv4)}/{expected_suv4}"
        )
    for report in successful:
        geometry = report.get("geometry", {})
        if geometry.get("input_voxels") != geometry.get("output_voxels"):
            raise ValueError(f"Geometry voxel mismatch for {report['patient_id']}")
        roundtrip = report.get("roundtrip", {})
        if (
            not roundtrip.get("equal")
            or roundtrip.get("dice") != 1.0
            or roundtrip.get("differing_voxels") != 0
        ):
            raise ValueError(f"Round-trip failure for {report['patient_id']}")
        for validator in ("dciodvfy", "dcentvfy"):
            result = report.get(validator, {})
            if result.get("returncode") != 0 or result.get("errors"):
                raise ValueError(f"{validator} failure for {report['patient_id']}")
        if not report.get("subset_ok"):
            raise ValueError(f"Subset failure for {report['patient_id']}")

    sop_uids = [report["sop_instance_uid"] for report in successful]
    series_uids = [report["series_instance_uid"] for report in successful]
    if len(sop_uids) != len(set(sop_uids)):
        raise ValueError("Duplicate DICOM SEG SOPInstanceUID values found")
    if len(series_uids) != len(set(series_uids)):
        raise ValueError("Duplicate DICOM SEG SeriesInstanceUID values found")
    return {
        "status": "ok",
        "native": len(native),
        "suv4": len(suv4),
        "empty_suv4": empty,
        "dicom_seg_objects": len(successful),
        "unique_sop_instance_uids": len(set(sop_uids)),
        "unique_series_instance_uids": len(set(series_uids)),
        "dciodvfy_errors": 0,
        "dcentvfy_errors": 0,
        "roundtrip_failures": 0,
        "subset_failures": 0,
    }


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "patient_id",
        "variant",
        "status",
        "output_path",
        "sop_instance_uid",
        "series_instance_uid",
        "input_voxels",
        "roundtrip_dice",
        "dciodvfy_errors",
        "dcentvfy_errors",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "patient_id": report.get("patient_id", ""),
                    "variant": report.get("variant", ""),
                    "status": report.get("status", ""),
                    "output_path": report.get("output_path", ""),
                    "sop_instance_uid": report.get("sop_instance_uid", ""),
                    "series_instance_uid": report.get("series_instance_uid", ""),
                    "input_voxels": report.get("geometry", {}).get("input_voxels", report.get("source_voxels", "")),
                    "roundtrip_dice": report.get("roundtrip", {}).get("dice", ""),
                    "dciodvfy_errors": len(report.get("dciodvfy", {}).get("errors", [])),
                    "dcentvfy_errors": len(report.get("dcentvfy", {}).get("errors", [])),
                    "error": report.get("error", ""),
                }
            )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--selection-csv", required=True, type=Path)
    prepare.add_argument("--qc-csv", required=True, type=Path)
    prepare.add_argument("--project-dir", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--source-commit", required=True)

    for command in ("convert-one", "validate-one"):
        one = subparsers.add_parser(command)
        one.add_argument("--tasks-file", required=True, type=Path)
        selection = one.add_mutually_exclusive_group(required=True)
        selection.add_argument("--task-index", type=int)
        selection.add_argument("--patient-id")
        one.add_argument("--variant", choices=("native", "suv4"))
        one.add_argument("--dicom3tools-sif", required=True, type=Path)
        one.add_argument("--singularity", default="singularity")
        if command == "convert-one":
            one.add_argument("--dcmqi-sif", required=True, type=Path)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--expected-json", required=True, type=Path)
    summarize.add_argument("--output-json", required=True, type=Path)
    summarize.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_export(
            selection_csv=args.selection_csv,
            qc_csv=args.qc_csv,
            project_dir=args.project_dir,
            output_root=args.output_root,
            source_commit=args.source_commit,
        )
    elif args.command in {"convert-one", "validate-one"}:
        if args.patient_id and not args.variant:
            raise ValueError("--variant is required with --patient-id")
        task = _select_task(
            args.tasks_file,
            task_index=args.task_index,
            patient_id=args.patient_id,
            variant=args.variant,
        )
        if args.command == "convert-one":
            result = convert_task(
                task,
                dcmqi_sif=args.dcmqi_sif,
                dicom3tools_sif=args.dicom3tools_sif,
                singularity=args.singularity,
            )
        else:
            result = validate_existing_task(
                task,
                dicom3tools_sif=args.dicom3tools_sif,
                singularity=args.singularity,
            )
    else:
        expected = json.loads(args.expected_json.read_text())
        report_dir = Path(expected["report_dir"])
        result = summarize_reports(
            report_dir=report_dir,
            expected_native=int(expected["expected_native"]),
            expected_suv4=int(expected["expected_suv4"]),
            expected_empty_suv4=list(expected["empty_suv4"]),
        )
        reports = [json.loads(path.read_text()) for path in sorted(report_dir.glob("*.json"))]
        _atomic_json(args.output_json, result)
        _write_summary_csv(args.output_csv, reports)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
