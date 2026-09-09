#!/usr/bin/env python3
"""Bind an OHIF/Orthanc DICOMweb smoke inventory to exact local SEG objects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import urllib.request

import pydicom


EXPECTED_OHIF_REFERENCE = (
    "ohif/app:v3.12.15@"
    "sha256:42e69dff6bab61463b9844b3d640dab8045bb6909095e5f89435441480cfd0b3"
)
EXPECTED_ORTHANC_REFERENCE = (
    "orthancteam/orthanc:26.8.2@"
    "sha256:9758c8702a89abece99fcfe6d5571d5eaae59587e8e1ce36b9aafc8d4f24457b"
)


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


def _dicom_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Accept": 'multipart/related; type="application/dicom"'},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    if content_type.lower().startswith("multipart/related"):
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + content_type.encode("ascii")
            + b"\r\nMIME-Version: 1.0\r\n\r\n"
            + body
        )
        parts = [part.get_payload(decode=True) for part in message.iter_parts()]
        if len(parts) != 1 or parts[0] is None:
            raise ValueError(f"Expected one DICOM WADO part from {url}")
        return parts[0]
    if content_type.split(";", 1)[0].strip().lower() in {
        "application/dicom",
        "application/octet-stream",
    }:
        return body
    raise ValueError(f"Unexpected WADO Content-Type {content_type!r} from {url}")


def _docker_inspect(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    values = json.loads(completed.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"Unexpected docker inspect result for {arguments[-1]}")
    return values[0]


def _inspect_container(container_name: str) -> dict[str, Any]:
    container = _docker_inspect(["container", "inspect", container_name])
    if container.get("State", {}).get("Running") is not True:
        raise ValueError(f"Container is not running: {container_name}")
    configured_reference = str(container.get("Config", {}).get("Image", ""))
    if "@sha256:" not in configured_reference:
        raise ValueError(f"Container image is not digest-pinned: {configured_reference}")
    image_id = str(container.get("Image", ""))
    image = _docker_inspect(["image", "inspect", image_id])
    repo_digests = sorted(str(value) for value in image.get("RepoDigests") or [])
    configured_digest = configured_reference.rsplit("@", 1)[1]
    if not any(value.endswith(f"@{configured_digest}") for value in repo_digests):
        raise ValueError(
            f"Running image does not expose configured digest {configured_digest}"
        )
    return {
        "container_name": container_name,
        "configured_reference": configured_reference,
        "image_id": image_id,
        "repo_digests": repo_digests,
        "running": True,
    }


def validate(
    *,
    source_dir: Path,
    seg_dir: Path,
    dicomweb_url: str,
    exporter_commit: str,
    ohif_container: str,
    orthanc_container: str,
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
    source_sop_instance_uids = sorted(
        str(item.SOPInstanceUID) for item in source_headers
    )
    if len(set(source_sop_instance_uids)) != len(source_sop_instance_uids):
        raise ValueError("Source PET contains duplicate SOP Instance UIDs")

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
                "path": path,
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

    expected_sops_by_series = {source_series_uid: source_sop_instance_uids} | {
        item["series_instance_uid"]: [item["sop_instance_uid"]]
        for item in seg_objects
    }
    actual_sops_by_series: dict[str, list[str]] = {}
    for series_uid, expected_sops in expected_sops_by_series.items():
        instances = _json_get(
            f"{base}/studies/{study_uid}/series/{series_uid}/instances"
        )
        actual_sops = sorted(str(_value(item, "00080018")) for item in instances)
        actual_sops_by_series[series_uid] = actual_sops
        if actual_sops != sorted(expected_sops):
            raise ValueError(
                "DICOMweb SOP Instance UID inventory differs for "
                f"{series_uid}: {actual_sops} != {sorted(expected_sops)}"
            )

    for item in seg_objects:
        payload = _dicom_get(
            f"{base}/studies/{study_uid}/series/{item['series_instance_uid']}"
            f"/instances/{item['sop_instance_uid']}"
        )
        wado_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != item["output_bytes"] or wado_sha256 != item["output_sha256"]:
            raise ValueError(
                f"WADO bytes differ from local {item['variant']} SEG object"
            )
        item["wado_bytes"] = len(payload)
        item["wado_sha256"] = wado_sha256
        del item["path"]

    ohif = _inspect_container(ohif_container)
    orthanc = _inspect_container(orthanc_container)
    if ohif["configured_reference"] != EXPECTED_OHIF_REFERENCE:
        raise ValueError("Running OHIF image differs from the pinned reference")
    if orthanc["configured_reference"] != EXPECTED_ORTHANC_REFERENCE:
        raise ValueError("Running Orthanc image differs from the pinned reference")

    return {
        "status": "dicomweb_ok_ui_not_exercised",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "browser_render": "not_run_no_controllable_browser",
        "patient_id": patient_ids.pop(),
        "exporter_commit": exporter_commit,
        "study_instance_uid": study_uid,
        "source_series_instance_uid": source_series_uid,
        "source_sop_instance_uids": source_sop_instance_uids,
        "instances": sum(actual_counts.values()),
        "patients": 1,
        "studies": 1,
        "series": len(actual_counts),
        "seg_objects": sorted(seg_objects, key=lambda item: item["variant"]),
        "series_inventory": inventory,
        "series_sop_instance_uids": actual_sops_by_series,
        "ohif": ohif,
        "orthanc": orthanc,
        "dicomweb_url": base,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--seg-dir", required=True, type=Path)
    parser.add_argument("--dicomweb-url", required=True)
    parser.add_argument("--exporter-commit", required=True)
    parser.add_argument("--ohif-container", default="lionz-ohif-smoke-ohif")
    parser.add_argument("--orthanc-container", default="lionz-ohif-smoke-orthanc")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    report = validate(
        source_dir=args.source_dir,
        seg_dir=args.seg_dir,
        dicomweb_url=args.dicomweb_url,
        exporter_commit=args.exporter_commit,
        ohif_container=args.ohif_container,
        orthanc_container=args.orthanc_container,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
