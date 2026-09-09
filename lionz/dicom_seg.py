"""Standards-oriented helpers for exporting LION masks as DICOM SEG."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import ExplicitVRLittleEndian, SegmentationStorage
import SimpleITK as sitk


@dataclass(frozen=True)
class GeometryReport:
    """Measurements proving that mask canonicalization preserved its content."""

    input_voxels: int
    output_voxels: int
    max_corner_distance_mm: float


@dataclass(frozen=True)
class SourceSeries:
    """Identifiers, ordered instances, and geometry for one source series."""

    directory: Path
    files: tuple[Path, ...]
    image: sitk.Image
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    sop_instance_uids: tuple[str, ...]


@dataclass(frozen=True)
class SegReport:
    """Independently parsed identity and reference facts from a DICOM SEG."""

    sop_instance_uid: str
    series_instance_uid: str
    frame_count: int
    referenced_sop_instance_uids: frozenset[str]


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


def _one_value(headers: list[Dataset], attribute: str) -> str:
    values = {
        str(getattr(header, attribute, "")).strip()
        for header in headers
    }
    values.discard("")
    if len(values) != 1:
        raise ValueError(
            f"Source DICOM must contain one consistent {attribute}, found "
            f"{sorted(values)}"
        )
    return values.pop()


def load_source_series(
    directory: str | Path,
    expected_series_uid: str | None = None,
) -> SourceSeries:
    """Load and validate exactly one PET DICOM series from a directory."""

    source_directory = Path(directory)
    discovered_files = tuple(sorted(source_directory.rglob("*.dcm")))
    if not discovered_files:
        raise ValueError(f"No DICOM files found in {source_directory}")

    headers = [
        pydicom.dcmread(path, stop_before_pixels=True)
        for path in discovered_files
    ]
    patient_id = _one_value(headers, "PatientID")
    study_uid = _one_value(headers, "StudyInstanceUID")
    series_uid = _one_value(headers, "SeriesInstanceUID")
    frame_uid = _one_value(headers, "FrameOfReferenceUID")
    modality = _one_value(headers, "Modality")
    if modality != "PT":
        raise ValueError(f"Source DICOM Modality must be PT, found {modality}")
    if expected_series_uid is not None and series_uid != expected_series_uid:
        raise ValueError(
            "Source SeriesInstanceUID does not match manifest: "
            f"{series_uid} != {expected_series_uid}"
        )

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source_directory))
    if not series_ids or series_uid not in series_ids:
        raise ValueError(
            f"GDCM could not identify source SeriesInstanceUID {series_uid}"
        )
    ordered_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(source_directory), series_uid
    )
    ordered_files = tuple(Path(name) for name in ordered_names)
    if len(ordered_files) != len(discovered_files):
        raise ValueError(
            "GDCM source instance count differs from discovered DICOM count: "
            f"{len(ordered_files)} != {len(discovered_files)}"
        )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in ordered_files])
    image = reader.Execute()
    header_by_path = {
        path.resolve(): header for path, header in zip(discovered_files, headers)
    }
    sop_uids = tuple(
        str(header_by_path[path.resolve()].SOPInstanceUID)
        for path in ordered_files
    )
    if len(set(sop_uids)) != len(sop_uids):
        raise ValueError("Source DICOM contains duplicate SOPInstanceUID values")

    return SourceSeries(
        directory=source_directory,
        files=ordered_files,
        image=image,
        patient_id=patient_id,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        frame_of_reference_uid=frame_uid,
        sop_instance_uids=sop_uids,
    )


def _referenced_sop_uids(dataset: Dataset) -> frozenset[str]:
    referenced: set[str] = set()
    for series in getattr(dataset, "ReferencedSeriesSequence", []):
        for instance in getattr(series, "ReferencedInstanceSequence", []):
            uid = getattr(instance, "ReferencedSOPInstanceUID", None)
            if uid:
                referenced.add(str(uid))
    for frame in getattr(dataset, "PerFrameFunctionalGroupsSequence", []):
        for derivation in getattr(frame, "DerivationImageSequence", []):
            for source in getattr(derivation, "SourceImageSequence", []):
                uid = getattr(source, "ReferencedSOPInstanceUID", None)
                if uid:
                    referenced.add(str(uid))
    return frozenset(referenced)


def normalize_seg_dataset(path: str | Path) -> tuple[str, ...]:
    """Apply lossless compatibility normalizations before SEG validation."""

    source_path = Path(path)
    dataset = pydicom.dcmread(source_path)
    changes: list[str] = []
    if (
        "ClinicalTrialSeriesID" in dataset
        and "ClinicalTrialCoordinatingCenterName" not in dataset
    ):
        dataset.ClinicalTrialCoordinatingCenterName = ""
        changes.append("added_empty_ClinicalTrialCoordinatingCenterName")
    if "SegmentsOverlap" in dataset:
        del dataset.SegmentsOverlap
        changes.append("removed_optional_SegmentsOverlap")
    if changes:
        temporary = source_path.with_name(f".{source_path.name}.normalize.tmp")
        dataset.save_as(temporary, enforce_file_format=True)
        temporary.replace(source_path)
    return tuple(changes)


def _code_tuple(dataset: Dataset) -> tuple[str, str, str]:
    return (
        str(dataset.CodeValue),
        str(dataset.CodingSchemeDesignator),
        str(dataset.CodeMeaning),
    )


def validate_seg_dataset(
    path: str | Path,
    source: SourceSeries,
    variant: str,
) -> SegReport:
    """Independently validate DICOM SEG identity, semantics, and references."""

    expected_labels = {
        "native": "LION FDG tumor",
        "suv4": "LION FDG tumor SUV>=4",
    }
    if variant not in expected_labels:
        raise ValueError(f"Unknown DICOM SEG variant: {variant}")
    dataset = pydicom.dcmread(path)

    if str(dataset.SOPClassUID) != str(SegmentationStorage):
        raise ValueError(f"Unexpected SOPClassUID: {dataset.SOPClassUID}")
    if dataset.Modality != "SEG" or dataset.SegmentationType != "BINARY":
        raise ValueError("Output must be a binary DICOM SEG object")
    if str(dataset.file_meta.TransferSyntaxUID) != str(ExplicitVRLittleEndian):
        raise ValueError(
            "DICOM SEG must use uncompressed Explicit VR Little Endian"
        )
    if dataset.PatientID != source.patient_id:
        raise ValueError("DICOM SEG PatientID differs from source")
    if str(dataset.StudyInstanceUID) != source.study_instance_uid:
        raise ValueError("DICOM SEG StudyInstanceUID differs from source")
    if str(dataset.FrameOfReferenceUID) != source.frame_of_reference_uid:
        raise ValueError("DICOM SEG FrameOfReferenceUID differs from source")

    referenced_series_uids = {
        str(item.SeriesInstanceUID)
        for item in getattr(dataset, "ReferencedSeriesSequence", [])
    }
    if referenced_series_uids != {source.series_instance_uid}:
        raise ValueError(
            "DICOM SEG does not reference the expected source SeriesInstanceUID"
        )
    referenced_sops = _referenced_sop_uids(dataset)
    if not referenced_sops:
        raise ValueError("DICOM SEG contains no source SOPInstanceUID references")
    expected_sops = frozenset(source.sop_instance_uids)
    unexpected_sops = referenced_sops.difference(expected_sops)
    if unexpected_sops:
        raise ValueError(
            "DICOM SEG references SOPInstanceUID values outside the source series: "
            f"{sorted(unexpected_sops)}"
        )
    missing_sops = expected_sops.difference(referenced_sops)
    if missing_sops:
        raise ValueError(
            "DICOM SEG is missing source SOPInstanceUID references: "
            f"{sorted(missing_sops)}"
        )

    if len(dataset.SegmentSequence) != 1:
        raise ValueError("DICOM SEG must contain exactly one segment")
    segment = dataset.SegmentSequence[0]
    if segment.SegmentLabel != expected_labels[variant]:
        raise ValueError(f"Unexpected SegmentLabel: {segment.SegmentLabel}")
    if segment.SegmentAlgorithmType != "AUTOMATIC":
        raise ValueError("SegmentAlgorithmType must be AUTOMATIC")
    if segment.SegmentAlgorithmName != "LION FDG 1.0.5":
        raise ValueError("Unexpected SegmentAlgorithmName")
    category = _code_tuple(segment.SegmentedPropertyCategoryCodeSequence[0])
    property_type = _code_tuple(segment.SegmentedPropertyTypeCodeSequence[0])
    if category != (
        "49755003",
        "SCT",
        "Morphologically Abnormal Structure",
    ):
        raise ValueError(f"Unexpected segment category code: {category}")
    if property_type != ("108369006", "SCT", "Neoplasm"):
        raise ValueError(f"Unexpected segment property type code: {property_type}")

    return SegReport(
        sop_instance_uid=str(dataset.SOPInstanceUID),
        series_instance_uid=str(dataset.SeriesInstanceUID),
        frame_count=int(dataset.NumberOfFrames),
        referenced_sop_instance_uids=referenced_sops,
    )


def _parse_validator_errors(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip().lower().startswith("error")
    )


def parse_dciodvfy(text: str) -> tuple[str, ...]:
    """Return fatal dciodvfy findings while leaving warnings for QC."""

    return _parse_validator_errors(text)


def parse_dcentvfy(text: str) -> tuple[str, ...]:
    """Return fatal dcentvfy findings while leaving warnings for QC."""

    return _parse_validator_errors(text)
