#!/usr/bin/env python3
"""Bind an OHIF/Orthanc DICOMweb smoke inventory to exact local SEG objects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import urllib.request

import pydicom


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(dataset: dict[str, Any], tag: str) -> Any:
    values = dataset.get(tag, {}).get("Value", [])
    return values[0] if values else None


def _json_get(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"Accept": "application/dicom+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def validate(
    *,
    source_dir: Path,
    seg_dir: Path,
    dicomweb_url: str,
    exporter_commit: str,
    ohif_image: str,
    ohif_digest: str,
    orthanc_image: str,
    orthanc_digest: str,
) -> dict[str, Any]:
    source_headers = [
        pydicom.dcmread(path, stop_before_pixels=True)
        for path in sorted(source_dir.glob("*.dcm"))
    ]
    if not source_headers:
        raise ValueError("No source PET DICOM files found")
    study_uids = {str(item.StudyInstanceUID) for item in source_headers}
    source_series_uids = {str(item.SeriesInstanceUID) for item in source_headers}
    patient_ids = {str(item.PatientID) for item in source_headers}
    if len(study_uids) != 1 or len(source_series_uids) != 1 or len(patient_ids) != 1:
        raise ValueError("Source PET is not one patient/study/series")
    study_uid = study_uids.pop()
    source_series_uid = source_series_uids.pop()

    variants_by_description = {
        "LION FDG native DICOM SEG": "native",
        "LION FDG SUV>=4 DICOM SEG": "suv4",
    }
    seg_objects: list[dict[str, Any]] = []
    for path in sorted(seg_dir.glob("*.dcm")):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        description = str(dataset.SeriesDescription)
        if description not in variants_by_description:
            raise ValueError(f"Unexpected SEG SeriesDescription: {description}")
        if str(dataset.StudyInstanceUID) != study_uid:
            raise ValueError("SEG and source StudyInstanceUID differ")
        seg_objects.append(
            {
                "variant": variants_by_description[description],
                "sop_instance_uid": str(dataset.SOPInstanceUID),
                "series_instance_uid": str(dataset.SeriesInstanceUID),
                "output_bytes": path.stat().st_size,
                "output_sha256": _sha256(path),
            }
        )
    if {item["variant"] for item in seg_objects} != {"native", "suv4"}:
        raise ValueError("SEG directory does not contain both variants")

    base = dicomweb_url.rstrip("/")
    studies = _json_get(f"{base}/studies")
    if [_value(item, "0020000D") for item in studies] != [study_uid]:
        raise ValueError("DICOMweb study inventory differs from smoke source")
    series = _json_get(f"{base}/studies/{study_uid}/series")
    inventory = [
        {
            "modality": _value(item, "00080060"),
            "description": _value(item, "0008103E"),
            "instances": int(_value(item, "00201209")),
            "series_instance_uid": _value(item, "0020000E"),
        }
        for item in series
    ]
    expected_counts = {source_series_uid: len(source_headers)} | {
        item["series_instance_uid"]: 1 for item in seg_objects
    }
    actual_counts = {
        str(item["series_instance_uid"]): int(item["instances"])
        for item in inventory
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"DICOMweb series inventory differs: {actual_counts} != {expected_counts}"
        )

    return {
        "status": "dicomweb_ok_ui_not_exercised",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "browser_render": "not_run_no_controllable_browser",
        "patient_id": patient_ids.pop(),
        "exporter_commit": exporter_commit,
        "study_instance_uid": study_uid,
        "source_series_instance_uid": source_series_uid,
        "instances": sum(actual_counts.values()),
        "patients": 1,
        "studies": 1,
        "series": len(actual_counts),
        "seg_objects": sorted(seg_objects, key=lambda item: item["variant"]),
        "series_inventory": inventory,
        "ohif": {"image": ohif_image, "digest": ohif_digest},
        "orthanc": {"image": orthanc_image, "digest": orthanc_digest},
        "dicomweb_url": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--seg-dir", required=True, type=Path)
    parser.add_argument("--dicomweb-url", required=True)
    parser.add_argument("--exporter-commit", required=True)
    parser.add_argument("--ohif-image", required=True)
    parser.add_argument("--ohif-digest", required=True)
    parser.add_argument("--orthanc-image", required=True)
    parser.add_argument("--orthanc-digest", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = validate(
        source_dir=args.source_dir,
        seg_dir=args.seg_dir,
        dicomweb_url=args.dicomweb_url,
        exporter_commit=args.exporter_commit,
        ohif_image=args.ohif_image,
        ohif_digest=args.ohif_digest,
        orthanc_image=args.orthanc_image,
        orthanc_digest=args.orthanc_digest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
