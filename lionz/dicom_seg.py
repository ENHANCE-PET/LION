"""Standards-oriented helpers for exporting LION masks as DICOM SEG."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any
import uuid

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class GeometryReport:
    """Measurements proving that mask canonicalization preserved its content."""

    input_voxels: int
    output_voxels: int
    max_corner_distance_mm: float


def _physical_corners(image: sitk.Image) -> np.ndarray:
    last_index = tuple(size - 1 for size in image.GetSize())
    corners = product(*((0, end) for end in last_index))
    return np.asarray(
        [image.TransformIndexToPhysicalPoint(index) for index in corners],
        dtype=np.float64,
    )


def physical_corner_distance(left: sitk.Image, right: sitk.Image) -> float:
    """Return the maximum nearest-corner distance between two image volumes."""

    if left.GetDimension() != right.GetDimension():
        raise ValueError("Images must have the same dimension")
    left_corners = _physical_corners(left)
    right_corners = _physical_corners(right)
    distances = np.linalg.norm(
        left_corners[:, np.newaxis, :] - right_corners[np.newaxis, :, :],
        axis=2,
    )
    return float(
        max(
            distances.min(axis=0).max(initial=0.0),
            distances.min(axis=1).max(initial=0.0),
        )
    )


def _binary_array(image: sitk.Image, *, name: str) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite binary values")
    values = np.unique(array)
    if not set(values.tolist()).issubset({0, 1}):
        raise ValueError(
            f"{name} must be binary with values {{0, 1}}, found {values.tolist()}"
        )
    return array


def canonicalize_binary_mask(
    mask: sitk.Image,
    reference: sitk.Image,
    tolerance_mm: float = 0.05,
) -> tuple[sitk.Image, GeometryReport]:
    """Resample a binary mask onto an exact reference grid in physical space."""

    if mask.GetDimension() != 3 or reference.GetDimension() != 3:
        raise ValueError("DICOM SEG export requires three-dimensional images")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be non-negative")

    input_array = _binary_array(mask, name="Mask")
    input_voxels = int(np.count_nonzero(input_array))
    corner_distance = physical_corner_distance(mask, reference)
    if corner_distance > tolerance_mm:
        raise ValueError(
            "Mask and source DICOM physical extent differ by "
            f"{corner_distance:.6f} mm (tolerance {tolerance_mm:.6f} mm)"
        )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    canonical = resampler.Execute(mask)

    output_array = _binary_array(canonical, name="Canonical mask")
    output_voxels = int(np.count_nonzero(output_array))
    if output_voxels != input_voxels:
        raise ValueError(
            "Canonicalization changed foreground voxel count from "
            f"{input_voxels} to {output_voxels}"
        )

    return canonical, GeometryReport(
        input_voxels=input_voxels,
        output_voxels=output_voxels,
        max_corner_distance_mm=corner_distance,
    )


def uid_from_key(key: str) -> str:
    """Create a deterministic globally unique DICOM UID for a stable key."""

    if not key:
        raise ValueError("UID key must not be empty")
    namespace_key = f"https://github.com/ENHANCE-PET/LION/dicom-seg/{key}"
    return f"2.25.{uuid.uuid5(uuid.NAMESPACE_URL, namespace_key).int}"


def build_dcmqi_metadata(
    variant: str,
    source_commit: str,
    tracking_uid: str,
) -> dict[str, Any]:
    """Build dcmqi metadata for one LION FDG neoplasm segment."""

    variants = {
        "native": {
            "series_description": "LION FDG native DICOM SEG",
            "series_number": "9001",
            "label": "LION FDG tumor",
            "description": "LION FDG native tumor segmentation",
            "color": [230, 25, 75],
        },
        "suv4": {
            "series_description": "LION FDG SUV>=4 DICOM SEG",
            "series_number": "9002",
            "label": "LION FDG tumor SUV>=4",
            "description": "LION FDG tumor segmentation thresholded at SUV 4",
            "color": [255, 215, 0],
        },
    }
    try:
        selected = variants[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown DICOM SEG variant: {variant}") from exc

    return {
        "@schema": (
            "https://raw.githubusercontent.com/qiicr/dcmqi/master/"
            "doc/schemas/seg-schema.json#"
        ),
        "ContentCreatorName": "LION^FDG",
        "ContentLabel": "LIONFDG",
        "ContentDescription": f"LION FDG 1.0.5 source {source_commit}",
        "SeriesDescription": selected["series_description"],
        "SeriesNumber": selected["series_number"],
        "InstanceNumber": "1",
        "segmentAttributes": [
            [
                {
                    "labelID": 1,
                    "SegmentDescription": selected["description"],
                    "SegmentLabel": selected["label"],
                    "SegmentedPropertyCategoryCodeSequence": {
                        "CodeValue": "49755003",
                        "CodingSchemeDesignator": "SCT",
                        "CodeMeaning": "Morphologically Abnormal Structure",
                    },
                    "SegmentedPropertyTypeCodeSequence": {
                        "CodeValue": "108369006",
                        "CodingSchemeDesignator": "SCT",
                        "CodeMeaning": "Neoplasm",
                    },
                    "SegmentAlgorithmType": "AUTOMATIC",
                    "SegmentAlgorithmName": "LION FDG 1.0.5",
                    "recommendedDisplayRGBValue": selected["color"],
                    "TrackingIdentifier": selected["label"],
                    "TrackingUniqueIdentifier": tracking_uid,
                }
            ]
        ],
    }
