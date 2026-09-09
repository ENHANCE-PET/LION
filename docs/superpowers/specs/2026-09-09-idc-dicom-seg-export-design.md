# IDC-Compatible LION DICOM SEG Export Design

## Purpose

Create standards-compliant DICOM Segmentation objects for the LION FDG masks
generated from the IDC `LUNG-PET-CT-Dx` collection. The export must reference
the original PET DICOM instances, preserve the mask exactly in patient space,
and pass structural, semantic, round-trip, and viewer-oriented validation.

## Scope

- Export one single-segment binary DICOM SEG object for every non-empty native
  and SUV>=4 mask.
- Produce 133 native objects and 130 SUV>=4 objects. The three empty SUV>=4
  masks (`Lung_Dx-A0204`, `Lung_Dx-A0232`, and `Lung_Dx-A0247`) are represented
  by explicit `NO_SEGMENT_ABOVE_THRESHOLD` QC records and no SEG object.
- Retain NIfTI masks as the research-format source of truth.
- Keep the original PET DICOM outside the deliverable archive; preserve a
  manifest that identifies every referenced source series.
- Do not add a generic image-series NIfTI-to-DICOM converter. This workflow
  creates DICOM SEG objects only.

## Conversion Architecture

The reusable conversion code has three boundaries:

1. **Geometry canonicalization** reads a source PET DICOM series and a binary
   NIfTI mask, verifies that their physical extents agree, and resamples the
   mask onto the exact DICOM grid with an identity transform and nearest-
   neighbour interpolation. It rejects non-binary inputs, mismatched Frames of
   Reference, out-of-bounds foreground, and voxel-count changes.
2. **DICOM SEG encoding** invokes the official QIICR `dcmqi`
   `itkimage2segimage` executable from a version- and digest-pinned container.
   It uses the canonical mask, the original PET DICOM directory, and validated
   coded metadata to create one binary SEG object per mask.
3. **Validation** treats every output as untrusted until it passes all gates in
   this document. A batch manifest records source identifiers, output UIDs,
   checksums, voxel counts, tool versions, and every validation result.

The LMU workflow downloads the pinned containers inside a Slurm job, stores
them under the existing `/data2` project, converts one smoke patient first,
and then converts the cohort in parallel on `jobs-cpu`.

## Geometry Rules

- Sort and load the source PET instances using their DICOM geometry, not file
  names.
- Assert one Patient ID, Study Instance UID, Series Instance UID, and Frame of
  Reference UID per source directory.
- Verify the source Series Instance UID against the IDC selection manifest.
- Compare the eight physical corner points of mask and source volumes with an
  absolute tolerance of 0.05 mm.
- Resample with an identity physical transform and nearest-neighbour
  interpolation onto the source PET size, spacing, origin, and direction.
- Write the canonical intermediate as an unsigned 8-bit NRRD label map.
- Require values to be exactly `{0, 1}` and require foreground voxel count to
  remain exactly unchanged after canonicalization.

This explicitly handles the observed LION NIfTI axis reversal while preserving
patient-space location.

## DICOM SEG Profile

- SOP Class: Segmentation Storage (`1.2.840.10008.5.1.4.1.1.66.4`).
- Segmentation Type: `BINARY`.
- One segment per object for broad viewer compatibility.
- Transfer syntax: uncompressed Explicit VR Little Endian.
- Source references: all required original PET SOP Instance UIDs, plus the
  source Study, Series, and Frame of Reference UIDs.
- Segment category/type: validated SNOMED CT concepts for a morphologically
  abnormal neoplastic structure.
- Algorithm type: `AUTOMATIC`.
- Algorithm name: `LION FDG`.
- Algorithm version: `1.0.5` plus the source commit recorded in provenance.
- Native label: `LION FDG tumor`.
- Thresholded label: `LION FDG tumor SUV>=4`.
- Series descriptions and series numbers distinguish native from SUV>=4.
- DICOM UIDs are unique, valid, and recorded in the output manifest.
- Patient identity and study identity are inherited from the source PET.

## Validation Gates

An object is accepted only when all applicable checks pass:

1. `dciodvfy` reports no errors for the individual object.
2. `dcentvfy` reports no errors for the SEG together with its referenced PET
   instances.
3. `pydicom` independently verifies the SEG SOP Class, binary type, one segment,
   source series/study/instance references, Frame of Reference, patient fields,
   algorithm metadata, and coded property metadata.
4. QIICR `segimage2itkimage` converts the SEG back to a label map.
5. The round-tripped label map equals the canonical input voxel-for-voxel:
   Dice = 1.0, zero differing voxels, and identical foreground count.
6. The SUV>=4 foreground is a subset of the matching native foreground.
7. Output SOP Instance UIDs and Series Instance UIDs are unique across the
   cohort.
8. A smoke case is opened with the source PET in both 3D Slicer and an OHIF
   DICOM SEG-capable viewer before full-batch acceptance.

Warnings from validators are retained in QC. Any validator error, missing
reference, geometry mismatch, round-trip mismatch, duplicate UID, unexpected
empty native mask, or unexpected non-empty/empty state fails the batch.

## Outputs

Remote outputs live under:

```text
/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/results/dicom-seg/
  native/
  suv4/
  canonical/
  metadata/
  qc/
```

The final archive contains the 263 SEG objects, the IDC selection manifest,
conversion metadata, checksums, validator output, round-trip metrics, tool and
container digests, Slurm scripts/logs, and explicit records for the three empty
SUV>=4 masks. It excludes raw DICOM and disposable canonical intermediates.

## Acceptance Criteria

- 133/133 native masks and 130/130 non-empty SUV>=4 masks produce DICOM SEG.
- All 263 objects pass every automated validation gate.
- All 263 round trips have Dice 1.0 and zero differing voxels.
- All source references resolve to the intended IDC PET series.
- The three empty SUV>=4 masks are present in QC with
  `NO_SEGMENT_ABOVE_THRESHOLD` and no fabricated SEG files.
- The smoke object displays aligned with PET in 3D Slicer and OHIF.
- A verified archive is copied to the local Mac with matching SHA-256 hashes.

No claim is made that viewers without DICOM SEG support can display the files.
The target is standards-compliant behavior in DICOM SEG-capable viewers.
