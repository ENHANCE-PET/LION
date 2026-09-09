#!/usr/bin/env python3
"""Headless 3D Slicer import check for a PET series and LION SEG objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

from DICOMLib import DICOMUtils
import slicer


def main() -> int:
    bundle = Path(os.environ["LIONZ_SLICER_BUNDLE"])
    report_path = Path(os.environ["LIONZ_SLICER_REPORT"])
    expected_labels = {"LION FDG tumor", "LION FDG tumor SUV>=4"}

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
