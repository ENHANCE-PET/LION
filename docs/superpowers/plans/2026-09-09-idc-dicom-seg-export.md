# IDC-Compatible LION DICOM SEG Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every non-empty LION FDG mask for the IDC `LUNG-PET-CT-Dx` cohort into a source-referenced, standards-compliant, independently validated DICOM SEG object.

**Architecture:** A small reusable Python module canonicalizes binary masks onto the exact source-DICOM grid and builds deterministic dcmqi metadata. A dataset CLI invokes pinned QIICR containers, validates each output structurally and by lossless round trip, and writes an auditable manifest. LMU Slurm jobs perform tool acquisition, one-case smoke validation, parallel batch conversion, and final archival.

**Tech Stack:** Python 3.10+, SimpleITK, NumPy, pydicom, QIICR dcmqi v1.5.7, dicom3tools (`dciodvfy`, `dcentvfy`), Singularity, Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-idc-dicom-seg-export-design.md`

## Global Constraints

- Produce one single-segment binary DICOM SEG object per non-empty mask.
- Produce exactly 133 native objects and 130 SUV>=4 objects.
- Record `NO_SEGMENT_ABOVE_THRESHOLD` and create no SEG for `Lung_Dx-A0204`, `Lung_Dx-A0232`, and `Lung_Dx-A0247` SUV>=4 masks.
- Canonicalize with identity physical-space resampling and nearest-neighbour interpolation only.
- Require binary `{0, 1}` values and exact foreground-voxel preservation.
- Use SCT `49755003` (`Morphologically Abnormal Structure`) as category and SCT `108369006` (`Neoplasm`) as type.
- Require zero `dciodvfy` errors, zero `dcentvfy` errors attributable to the SEG/reference relationship, and exact dcmqi round-trip equality.
- Download containers only inside a Slurm job and store pinned SIF files plus SHA-256 hashes under `/data2`.
- Do not include raw DICOM or canonical intermediates in the final archive.

---

### Task 1: Geometry canonicalization and coded metadata

**Files:**
- Create: `lionz/dicom_seg.py`
- Create: `tests/test_dicom_seg.py`

**Interfaces:**
- Produces: `canonicalize_binary_mask(mask, reference, tolerance_mm) -> (image, GeometryReport)`
- Produces: `physical_corner_distance(left, right) -> float`
- Produces: `build_dcmqi_metadata(variant, source_commit, tracking_uid) -> dict`
- Produces: `uid_from_key(key) -> str`

- [ ] **Step 1: Write failing geometry and metadata tests**

```python
def test_canonicalization_preserves_axis_reversed_mask_voxels():
    reference = make_reference_image()
    mask = make_physically_equal_y_reversed_mask(reference)
    canonical, report = canonicalize_binary_mask(mask, reference)
    assert canonical.GetDirection() == reference.GetDirection()
    assert report.input_voxels == report.output_voxels == 2
    assert np.array_equal(sitk.GetArrayFromImage(canonical), EXPECTED_ARRAY)

def test_canonicalization_rejects_nonbinary_mask():
    with pytest.raises(ValueError, match="binary"):
        canonicalize_binary_mask(mask_with_value_2(), make_reference_image())

def test_metadata_uses_standard_neoplasm_codes():
    metadata = build_dcmqi_metadata("native", "abc123", uid_from_key("case:native"))
    segment = metadata["segmentAttributes"][0][0]
    assert segment["SegmentedPropertyCategoryCodeSequence"]["CodeValue"] == "49755003"
    assert segment["SegmentedPropertyTypeCodeSequence"]["CodeValue"] == "108369006"
    assert segment["SegmentAlgorithmType"] == "AUTOMATIC"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest tests/test_dicom_seg.py -v`

Expected: collection/import failure because `lionz.dicom_seg` does not exist.

- [ ] **Step 3: Implement the minimal geometry and metadata module**

Implement immutable reports, eight-corner physical matching, identity nearest-
neighbour resampling to the reference grid, strict binary/count checks,
deterministic `2.25` UUID-derived tracking UIDs, and dcmqi JSON dictionaries for
`native` and `suv4`.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_dicom_seg.py -v && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the tested module**

```bash
git add lionz/dicom_seg.py tests/test_dicom_seg.py
git commit -m "feat: canonicalize masks for DICOM SEG export"
```

### Task 2: Source-series and DICOM SEG validation

**Files:**
- Modify: `lionz/dicom_seg.py`
- Modify: `tests/test_dicom_seg.py`

**Interfaces:**
- Produces: `load_source_series(path, expected_series_uid) -> SourceSeries`
- Produces: `validate_seg_dataset(path, source, variant) -> SegReport`
- Produces: `parse_dciodvfy(text) -> tuple[str, ...]`
- Produces: `parse_dcentvfy(text) -> tuple[str, ...]`

- [ ] **Step 1: Add failing tests for UID consistency and validator errors**

```python
def test_source_loader_rejects_wrong_manifest_series_uid(tmp_path):
    write_test_dicom_series(tmp_path, series_uid="1.2.3")
    with pytest.raises(ValueError, match="SeriesInstanceUID"):
        load_source_series(tmp_path, expected_series_uid="1.2.4")

def test_validator_parser_treats_errors_as_fatal_but_retains_warnings():
    assert parse_dciodvfy("Warning - retained\n") == ()
    assert parse_dciodvfy("Error - missing attribute\n") == ("Error - missing attribute",)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest tests/test_dicom_seg.py -v`

Expected: failures naming the missing loader and validator functions.

- [ ] **Step 3: Implement source and SEG validation**

Load source instances with pydicom and SimpleITK/GDCM ordering; require one
patient, study, series, and Frame of Reference. Validate SEG SOP class, binary
type, one segment, coded metadata, automatic algorithm identity, patient/study/
frame identity, referenced source series and SOP instances, and unique output
UIDs. Parse validator output by complete lines beginning with `Error`.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_dicom_seg.py -v && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit source/reference validation**

```bash
git add lionz/dicom_seg.py tests/test_dicom_seg.py
git commit -m "feat: validate DICOM SEG source references"
```

### Task 3: Dataset conversion CLI and behavior tests

**Files:**
- Create: `scripts/export_idc_dicom_seg.py`
- Create: `tests/test_export_idc_dicom_seg.py`

**Interfaces:**
- Consumes: the Task 1 and Task 2 APIs.
- Produces: CLI subcommands `prepare`, `convert-one`, `validate-one`, and `summarize`.
- Produces: per-object JSON reports and `dicom_seg_validation.csv/json`.

- [ ] **Step 1: Write failing CLI behavior tests**

```python
def test_prepare_records_empty_suv4_without_conversion(tmp_path):
    result = run_prepare(selection_csv, qc_csv_with_three_empty_masks, tmp_path)
    assert result["expected_native"] == 133
    assert result["expected_suv4"] == 130
    assert result["empty_suv4"] == ["Lung_Dx-A0204", "Lung_Dx-A0232", "Lung_Dx-A0247"]

def test_summarize_rejects_any_failed_validator(tmp_path):
    write_report(tmp_path, dciodvfy_errors=["Error - bad reference"])
    with pytest.raises(SystemExit):
        summarize(tmp_path, expected_native=1, expected_suv4=0)
```

- [ ] **Step 2: Run the CLI tests and confirm RED**

Run: `uv run pytest tests/test_export_idc_dicom_seg.py -v`

Expected: failure because the script does not exist.

- [ ] **Step 3: Implement the CLI**

Use subprocess argument lists without shell expansion. `convert-one` writes a
canonical NRRD and metadata JSON, invokes `itkimage2segimage`, runs `dciodvfy`
and `dcentvfy`, runs `segimage2itkimage`, compares the round trip exactly, and
writes an atomic JSON report. `summarize` rejects count, UID, reference,
geometry, validator, subset, or round-trip failures and records the three
expected empty SUV>=4 outcomes.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_export_idc_dicom_seg.py -v && uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the CLI**

```bash
git add scripts/export_idc_dicom_seg.py tests/test_export_idc_dicom_seg.py
git commit -m "feat: add validated IDC DICOM SEG export CLI"
```

### Task 4: LMU Slurm tool bootstrap, smoke test, and cohort run

**Files:**
- Create: `scripts/lmu/prepare_dicom_seg_tools.sbatch`
- Create: `scripts/lmu/export_idc_dicom_seg.sbatch`

**Interfaces:**
- Consumes: repository CLI and the existing `/data2` IDC project.
- Produces: pinned dcmqi/dicom3tools SIFs, smoke reports, 263 SEG objects, and cohort QC.

- [ ] **Step 1: Add shell syntax checks before implementation**

Run: `test ! -e scripts/lmu/prepare_dicom_seg_tools.sbatch && test ! -e scripts/lmu/export_idc_dicom_seg.sbatch`

Expected: success, proving the scripts are not pre-existing untested behavior.

- [ ] **Step 2: Write Slurm scripts with strict failure behavior**

The bootstrap job pulls `qiicr/dcmqi:v1.5.7` and `qiicr/dicom3tools:latest`
inside `jobs-cpu`, stores SIF SHA-256 hashes, and prints tool versions. The
export job validates prerequisites, runs `Lung_Dx-A0164` as a smoke case,
requires all validation gates, then fans out patient/variant tasks across CPU
workers and runs the strict summary gate.

- [ ] **Step 3: Verify script syntax and commit**

Run: `bash -n scripts/lmu/prepare_dicom_seg_tools.sbatch scripts/lmu/export_idc_dicom_seg.sbatch`

Expected: exit 0.

```bash
git add scripts/lmu/prepare_dicom_seg_tools.sbatch scripts/lmu/export_idc_dicom_seg.sbatch
git commit -m "ops: add LMU DICOM SEG export jobs"
```

- [ ] **Step 4: Copy the branch snapshot to `/data2` and run bootstrap**

Run the bootstrap with `sbatch --wait`, then independently inspect SIF hashes
and `itkimage2segimage`, `segimage2itkimage`, `dciodvfy`, and `dcentvfy`
versions.

- [ ] **Step 5: Run and inspect the one-patient smoke test**

Require zero validator errors, correct source UID references, and Dice 1.0 for
both native and SUV>=4 before the cohort phase begins.

- [ ] **Step 6: Run the full cohort and inspect fresh Slurm accounting**

Require exactly 133 native and 130 SUV>=4 objects, three expected empty records,
263 unique SOP UIDs, 263 unique Series UIDs, and zero failures.

### Task 5: Viewer smoke, archive, and repository integration

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-09-09-idc-dicom-seg-export.md`

**Interfaces:**
- Consumes: validated cohort and smoke outputs.
- Produces: viewer evidence, final checksummed archive, merged repository change.

- [ ] **Step 1: Validate a smoke case in two independent viewer stacks**

Load the source PET plus native and SUV>=4 SEG in 3D Slicer and an OHIF DICOM
SEG-capable deployment. Record screenshots/log evidence and confirm spatial
alignment and both display labels.

- [ ] **Step 2: Add concise user documentation and verify it**

Document the CLI inputs, dcmqi/dicom3tools requirements, empty-mask behavior,
and validation gates. Run `git diff --check` and the full test suite.

- [ ] **Step 3: Create and validate the final archive on Slurm**

Archive the 263 SEG objects, manifests, QC, metadata, tool hashes, scripts, and
logs. Verify archive CRC, file counts, and SHA-256 remotely; copy it to the
Mac and verify the same checksum locally.

- [ ] **Step 4: Complete repository review and integration**

Run the complete test suite and syntax checks, inspect the diff, push the
feature branch, create a PR, obtain review, merge, and confirm `main` equals
`origin/main` without touching unrelated root-worktree files.
