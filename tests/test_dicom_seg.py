import numpy as np
import pytest
import SimpleITK as sitk

from lionz.dicom_seg import (
    build_dcmqi_metadata,
    canonicalize_binary_mask,
    physical_corner_distance,
    uid_from_key,
)


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
