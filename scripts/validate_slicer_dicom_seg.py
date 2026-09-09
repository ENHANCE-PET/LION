#!/usr/bin/env python3
"""Headless 3D Slicer import check for a PET series and LION SEG objects."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback

from DICOMLib import DICOMUtils
import pydicom
import slicer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dicom_evidence(bundle: Path) -> tuple[str, str, list[dict[str, object]]]:
    source_headers = [
        pydicom.dcmread(path, stop_before_pixels=True)
        for path in sorted((bundle / "source-pt").glob("*.dcm"))
    ]
    if not source_headers:
        raise RuntimeError("Slicer smoke bundle contains no source PET DICOM")
    source_series_uids = {str(item.SeriesInstanceUID) for item in source_headers}
    study_uids = {str(item.StudyInstanceUID) for item in source_headers}
    if len(source_series_uids) != 1 or len(study_uids) != 1:
        raise RuntimeError("Slicer smoke source is not one DICOM study/series")

    variants_by_description = {
        "LION FDG native DICOM SEG": "native",
        "LION FDG SUV>=4 DICOM SEG": "suv4",
    }
    seg_objects: list[dict[str, object]] = []
    for path in sorted((bundle / "seg").glob("*.dcm")):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        description = str(dataset.SeriesDescription)
        if description not in variants_by_description:
            raise RuntimeError(f"Unexpected SEG SeriesDescription: {description}")
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
        raise RuntimeError("Slicer smoke bundle does not contain both SEG variants")
    return study_uids.pop(), source_series_uids.pop(), seg_objects


def main() -> int:
    bundle = Path(os.environ["LIONZ_SLICER_BUNDLE"])
    report_path = Path(os.environ["LIONZ_SLICER_REPORT"])
    exporter_commit = os.environ["LIONZ_EXPORTER_COMMIT"].strip()
    if len(exporter_commit) != 40:
        raise RuntimeError("LIONZ_EXPORTER_COMMIT must be a full Git commit")
    expected_labels = {"LION FDG tumor", "LION FDG tumor SUV>=4"}
    study_uid, source_series_uid, seg_objects = _dicom_evidence(bundle)

    slicer.mrmlScene.Clear(0)
    with DICOMUtils.TemporaryDICOMDatabase() as database:
        for directory_name in ("source-pt", "seg"):
            dicom_directory = bundle / directory_name
            if not dicom_directory.is_dir():
                raise RuntimeError(f"Missing Slicer smoke directory: {dicom_directory}")
            DICOMUtils.importDicom(str(dicom_directory), database)
        patient_uids = database.patients()
        if len(patient_uids) != 1:
            raise RuntimeError(f"Expected one Slicer DICOM patient, found {len(patient_uids)}")
        loaded_node_ids = DICOMUtils.loadPatientByUID(patient_uids[0])
        if not loaded_node_ids:
            raise RuntimeError("Slicer did not load any DICOM nodes")

        loaded_nodes = [
            slicer.mrmlScene.GetNodeByID(node_id) for node_id in loaded_node_ids
        ]
        loaded_nodes = [node for node in loaded_nodes if node is not None]
        segmentation_nodes = [
            node for node in loaded_nodes if node.IsA("vtkMRMLSegmentationNode")
        ]
        volume_nodes = [
            node for node in loaded_nodes if node.IsA("vtkMRMLScalarVolumeNode")
        ]
        if len(volume_nodes) != 1:
            raise RuntimeError(
                f"Expected one loaded PET volume node, found {len(volume_nodes)}"
            )
        reference_volume = volume_nodes[0]
        volume_bounds = [0.0] * 6
        reference_volume.GetRASBounds(volume_bounds)

        labels: list[str] = []
        segment_counts: list[int] = []
        reference_volume_ids: list[str] = []
        geometry_parameters: list[bool] = []
        bounds_within_reference: list[bool] = []
        nonempty_binary_labelmaps: list[bool] = []
        for node in segmentation_nodes:
            segmentation = node.GetSegmentation()
            segment_counts.append(segmentation.GetNumberOfSegments())
            referenced_volume = node.GetNodeReference(
                node.GetReferenceImageGeometryReferenceRole()
            )
            if referenced_volume is None:
                raise RuntimeError(f"SEG node {node.GetName()} has no referenced PET volume")
            reference_volume_ids.append(referenced_volume.GetID())
            geometry_parameters.append(
                bool(segmentation.GetConversionParameter("Reference image geometry"))
            )
            segmentation_bounds = [0.0] * 6
            node.GetRASBounds(segmentation_bounds)
            bounds_within_reference.append(
                all(
                    segmentation_bounds[2 * axis] >= volume_bounds[2 * axis] - 1e-3
                    and segmentation_bounds[2 * axis + 1]
                    <= volume_bounds[2 * axis + 1] + 1e-3
                    for axis in range(3)
                )
            )
            for index in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(index)
                segment = segmentation.GetSegment(segment_id)
                labels.append(segment.GetName())
                binary_labelmap = segment.GetRepresentation("Binary labelmap")
                scalars = (
                    binary_labelmap.GetPointData().GetScalars()
                    if binary_labelmap is not None
                    else None
                )
                nonempty_binary_labelmaps.append(
                    scalars is not None and scalars.GetRange()[1] > 0
                )

        if len(segmentation_nodes) != 2:
            raise RuntimeError(
                f"Expected two Slicer segmentation nodes, found {len(segmentation_nodes)}"
            )
        if segment_counts != [1, 1]:
            raise RuntimeError(f"Expected one segment per SEG object, found {segment_counts}")
        if set(labels) != expected_labels:
            raise RuntimeError(f"Unexpected Slicer segment labels: {labels}")
        if set(reference_volume_ids) != {reference_volume.GetID()}:
            raise RuntimeError(
                "Slicer SEG nodes are not both linked to the loaded PET volume"
            )
        if not all(geometry_parameters):
            raise RuntimeError("A Slicer SEG node is missing reference image geometry")
        if not all(bounds_within_reference):
            raise RuntimeError("A Slicer SEG extends outside the referenced PET bounds")
        if not all(nonempty_binary_labelmaps):
            raise RuntimeError("A Slicer SEG contains an empty binary labelmap")

        report = {
            "status": "ok",
            "slicer_version": slicer.app.applicationVersion,
            "patient_uids": patient_uids,
            "loaded_nodes": len(loaded_node_ids),
            "volume_nodes": len(volume_nodes),
            "segmentation_nodes": len(segmentation_nodes),
            "segments_per_node": segment_counts,
            "segment_labels": sorted(labels),
            "scene_cleared_before_load": True,
            "reference_volume_id": reference_volume.GetID(),
            "segmentation_reference_volume_ids": reference_volume_ids,
            "reference_geometry_present": all(geometry_parameters),
            "bounds_within_reference": all(bounds_within_reference),
            "nonempty_binary_labelmaps": all(nonempty_binary_labelmaps),
            "exporter_commit": exporter_commit,
            "study_instance_uid": study_uid,
            "source_series_instance_uid": source_series_uid,
            "seg_objects": sorted(seg_objects, key=lambda item: str(item["variant"])),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


try:
    exit_code = main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    slicer.util.exit(exit_code)
