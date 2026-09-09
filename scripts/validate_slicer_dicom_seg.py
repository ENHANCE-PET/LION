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

    with DICOMUtils.TemporaryDICOMDatabase() as database:
        DICOMUtils.importDicom(str(bundle), database)
        patient_uids = database.patients()
        if len(patient_uids) != 1:
            raise RuntimeError(f"Expected one Slicer DICOM patient, found {len(patient_uids)}")
        loaded_node_ids = DICOMUtils.loadPatientByUID(patient_uids[0])
        if not loaded_node_ids:
            raise RuntimeError("Slicer did not load any DICOM nodes")

        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        volume_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        labels: list[str] = []
        segment_counts: list[int] = []
        for node in segmentation_nodes:
            segmentation = node.GetSegmentation()
            segment_counts.append(segmentation.GetNumberOfSegments())
            for index in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(index)
                labels.append(segmentation.GetSegment(segment_id).GetName())

        if len(segmentation_nodes) != 2:
            raise RuntimeError(
                f"Expected two Slicer segmentation nodes, found {len(segmentation_nodes)}"
            )
        if segment_counts != [1, 1]:
            raise RuntimeError(f"Expected one segment per SEG object, found {segment_counts}")
        if set(labels) != expected_labels:
            raise RuntimeError(f"Unexpected Slicer segment labels: {labels}")
        if not volume_nodes:
            raise RuntimeError("Slicer did not load the referenced PET volume")

        report = {
            "status": "ok",
            "slicer_version": slicer.app.applicationVersion,
            "patient_uids": patient_uids,
            "loaded_nodes": len(loaded_node_ids),
            "volume_nodes": len(volume_nodes),
            "segmentation_nodes": len(segmentation_nodes),
            "segments_per_node": segment_counts,
            "segment_labels": sorted(labels),
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
