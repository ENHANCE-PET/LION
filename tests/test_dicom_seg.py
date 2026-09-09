import numpy as np
from pathlib import Path
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    SegmentationStorage,
    generate_uid,
)
import pytest
import SimpleITK as sitk
import subprocess
import sys

from lionz.dicom_seg import (
    SourceSeries,
    build_dcmqi_metadata,
    canonicalize_binary_mask,
    load_source_series,
    parse_dcentvfy,
    parse_dciodvfy,
    physical_corner_distance,
    uid_from_key,
    validate_seg_dataset,
)


def test_dicom_seg_import_does_not_load_inference_dependencies():
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lionz.dicom_seg; assert 'cv2' not in sys.modules",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _reference_image() -> sitk.Image:
    image = sitk.Image([3, 4, 2], sitk.sitkUInt8)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((-5.0, -7.0, 11.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _axis_reversed_mask(reference: sitk.Image) -> tuple[sitk.Image, np.ndarray]:
    expected = np.zeros((2, 4, 3), dtype=np.uint8)
    expected[0, 0, 0] = 1
    expected[1, 2, 2] = 1

    reversed_array = np.flip(expected, axis=1).copy()
    mask = sitk.GetImageFromArray(reversed_array)
    mask.SetSpacing(reference.GetSpacing())
    mask.SetOrigin(
        (
            reference.GetOrigin()[0],
            reference.GetOrigin()[1]
            + (reference.GetSize()[1] - 1) * reference.GetSpacing()[1],
            reference.GetOrigin()[2],
        )
    )
    mask.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    return mask, expected


def test_canonicalization_preserves_axis_reversed_mask_voxels():
    reference = _reference_image()
    mask, expected = _axis_reversed_mask(reference)

    canonical, report = canonicalize_binary_mask(mask, reference)

    assert canonical.GetSize() == reference.GetSize()
    assert canonical.GetSpacing() == reference.GetSpacing()
    assert canonical.GetOrigin() == reference.GetOrigin()
    assert canonical.GetDirection() == reference.GetDirection()
    assert report.input_voxels == report.output_voxels == 2
    assert report.max_corner_distance_mm == pytest.approx(0.0)
    assert np.array_equal(sitk.GetArrayFromImage(canonical), expected)


def test_corner_distance_is_independent_of_axis_order():
    reference = _reference_image()
    mask, _ = _axis_reversed_mask(reference)

    assert physical_corner_distance(mask, reference) == pytest.approx(0.0)


def test_canonicalization_rejects_nonbinary_mask():
    reference = _reference_image()
    mask = sitk.Image(reference)
    mask.SetPixel(0, 0, 0, 2)

    with pytest.raises(ValueError, match="binary"):
        canonicalize_binary_mask(mask, reference)


def test_canonicalization_rejects_different_physical_extent():
    reference = _reference_image()
    mask = sitk.Image(reference)
    mask.SetOrigin((-5.0, -7.0, 12.0))

    with pytest.raises(ValueError, match="physical extent"):
        canonicalize_binary_mask(mask, reference, tolerance_mm=0.05)


@pytest.mark.parametrize(
    ("variant", "label", "description", "color"),
    [
        ("native", "LION FDG tumor", "LION FDG native tumor segmentation", [230, 25, 75]),
        ("suv4", "LION FDG tumor SUV>=4", "LION FDG tumor segmentation thresholded at SUV 4", [255, 215, 0]),
    ],
)
def test_metadata_uses_standard_neoplasm_codes(
    variant: str,
    label: str,
    description: str,
    color: list[int],
):
    tracking_uid = uid_from_key(f"Lung_Dx-A0164:{variant}")

    metadata = build_dcmqi_metadata(variant, "39a2139", tracking_uid)

    segment = metadata["segmentAttributes"][0][0]
    assert segment["SegmentedPropertyCategoryCodeSequence"] == {
        "CodeValue": "49755003",
        "CodingSchemeDesignator": "SCT",
        "CodeMeaning": "Morphologically Abnormal Structure",
    }
    assert segment["SegmentedPropertyTypeCodeSequence"] == {
        "CodeValue": "108369006",
        "CodingSchemeDesignator": "SCT",
        "CodeMeaning": "Neoplasm",
    }
    assert segment["SegmentAlgorithmType"] == "AUTOMATIC"
    assert segment["SegmentAlgorithmName"] == "LION FDG 1.0.5"
    assert segment["SegmentLabel"] == label
    assert segment["SegmentDescription"] == description
    assert segment["recommendedDisplayRGBValue"] == color
    assert segment["TrackingUniqueIdentifier"] == tracking_uid
    assert metadata["ContentLabel"] == "LIONFDG"
    assert metadata["ContentDescription"] == "LION FDG 1.0.5 source 39a2139"
    assert metadata["SeriesNumber"] in {"9001", "9002"}


def test_uid_from_key_is_deterministic_and_valid():
    first = uid_from_key("Lung_Dx-A0164:native")

    assert first == uid_from_key("Lung_Dx-A0164:native")
    assert first != uid_from_key("Lung_Dx-A0164:suv4")
    assert first.startswith("2.25.")
    assert len(first) <= 64
    assert all(part.isdigit() for part in first.split("."))


def test_metadata_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        build_dcmqi_metadata("invalid", "39a2139", uid_from_key("invalid"))


def _write_pet_dicom_series(directory, *, series_uid: str) -> SourceSeries:
    patient_id = "Lung_Dx-TEST"
    study_uid = generate_uid()
    frame_uid = generate_uid()
    sop_uids = []
    for index in range(2):
        sop_uid = generate_uid()
        sop_uids.append(sop_uid)
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = PositronEmissionTomographyImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(
            directory / f"slice-{index}.dcm",
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        dataset.SOPClassUID = PositronEmissionTomographyImageStorage
        dataset.SOPInstanceUID = sop_uid
        dataset.PatientID = patient_id
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.Modality = "PT"
        dataset.SeriesNumber = 1
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(index)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0
        dataset.SpacingBetweenSlices = 1.0
        dataset.Rows = 2
        dataset.Columns = 2
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleIntercept = 0
        dataset.RescaleSlope = 1
        dataset.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
        dataset.save_as(directory / f"slice-{index}.dcm", enforce_file_format=True)

    return SourceSeries(
        directory=directory,
        files=tuple(directory / f"slice-{index}.dcm" for index in range(2)),
        image=sitk.Image([2, 2, 2], sitk.sitkUInt16),
        patient_id=patient_id,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        frame_of_reference_uid=frame_uid,
        sop_instance_uids=tuple(sop_uids),
    )


def test_source_loader_rejects_wrong_manifest_series_uid(tmp_path):
    _write_pet_dicom_series(tmp_path, series_uid="1.2.3")

    with pytest.raises(ValueError, match="SeriesInstanceUID"):
        load_source_series(tmp_path, expected_series_uid="1.2.4")


def test_source_loader_orders_and_identifies_one_pet_series(tmp_path):
    expected = _write_pet_dicom_series(tmp_path, series_uid="1.2.3")

    source = load_source_series(tmp_path, expected_series_uid="1.2.3")

    assert source.patient_id == expected.patient_id
    assert source.study_instance_uid == expected.study_instance_uid
    assert source.series_instance_uid == "1.2.3"
    assert source.frame_of_reference_uid == expected.frame_of_reference_uid
    assert source.image.GetSize() == (2, 2, 2)
    assert set(source.sop_instance_uids) == set(expected.sop_instance_uids)


def _code(value: str, meaning: str) -> Dataset:
    code = Dataset()
    code.CodeValue = value
    code.CodingSchemeDesignator = "SCT"
    code.CodeMeaning = meaning
    return code


def _write_seg_dataset(path, source: SourceSeries, *, variant: str = "native"):
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SegmentationStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = source.patient_id
    dataset.StudyInstanceUID = source.study_instance_uid
    dataset.FrameOfReferenceUID = source.frame_of_reference_uid
    dataset.Modality = "SEG"
    dataset.SegmentationType = "BINARY"
    dataset.NumberOfFrames = 1

    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = (
        "LION FDG tumor" if variant == "native" else "LION FDG tumor SUV>=4"
    )
    segment.SegmentAlgorithmType = "AUTOMATIC"
    segment.SegmentAlgorithmName = "LION FDG 1.0.5"
    segment.SegmentedPropertyCategoryCodeSequence = Sequence(
        [_code("49755003", "Morphologically Abnormal Structure")]
    )
    segment.SegmentedPropertyTypeCodeSequence = Sequence(
        [_code("108369006", "Neoplasm")]
    )
    dataset.SegmentSequence = Sequence([segment])

    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = source.series_instance_uid
    referenced_series.ReferencedInstanceSequence = Sequence([])
    for source_sop_uid in source.sop_instance_uids:
        item = Dataset()
        item.ReferencedSOPClassUID = PositronEmissionTomographyImageStorage
        item.ReferencedSOPInstanceUID = source_sop_uid
        referenced_series.ReferencedInstanceSequence.append(item)
    dataset.ReferencedSeriesSequence = Sequence([referenced_series])
    dataset.save_as(path, enforce_file_format=True)


def test_seg_validator_accepts_correct_source_references(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_pet_dicom_series(source_dir, series_uid="1.2.3")
    seg_path = tmp_path / "native.dcm"
    _write_seg_dataset(seg_path, source)

    report = validate_seg_dataset(seg_path, source, "native")

    assert report.referenced_sop_instance_uids == frozenset(
        source.sop_instance_uids
    )
    assert report.frame_count == 1


def test_seg_validator_rejects_wrong_source_series(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_pet_dicom_series(source_dir, series_uid="1.2.3")
    seg_path = tmp_path / "native.dcm"
    _write_seg_dataset(seg_path, source)
    dataset = __import__("pydicom").dcmread(seg_path)
    dataset.ReferencedSeriesSequence[0].SeriesInstanceUID = "1.2.4"
    dataset.save_as(seg_path, enforce_file_format=True)

    with pytest.raises(ValueError, match="source SeriesInstanceUID"):
        validate_seg_dataset(seg_path, source, "native")


def test_validator_parsers_treat_errors_as_fatal_but_retain_warnings():
    text = "Warning - retained for QC\nError - missing attribute\n"

    assert parse_dciodvfy(text) == ("Error - missing attribute",)
    assert parse_dcentvfy("Warning - retained for QC\n") == ()
