#!/usr/bin/env bash
#SBATCH --job-name=lionz-idc-seg-archive
#SBATCH --account=core-rad-digitx
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_archive_%j.out
#SBATCH --error=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_archive_%j.err

set -euo pipefail
umask 002

ROOT=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx
OUTPUT_ROOT="${ROOT}/results/dicom-seg"
QC_ROOT="${OUTPUT_ROOT}/qc"
ARCHIVE=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx-dicom-seg-2026-09-09.zip
TEMP_DIR=$(mktemp -d "${ROOT}/tmp/dicom-seg-archive.XXXXXX")
TEMP_ARCHIVE="${TEMP_DIR}/$(basename "${ARCHIVE}")"

cleanup() {
  rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

source "${ROOT}/runtime.env"
test -f "${QC_ROOT}/COHORT_VALIDATED"
test -s "${QC_ROOT}/dicom_seg_validation.json"
test -s "${QC_ROOT}/viewer/slicer_validation.json"
test -s "${QC_ROOT}/viewer/ohif_dicomweb_validation.json"
test -s "${ROOT}/manifest/fdg_pet_selection.csv"
test -s "${ROOT}/qc/segmentation_validation.csv"
test -s "${ROOT}/software/dicom-seg-tools/SHA256SUMS"
test -s "${ROOT}/software/dicom-seg-tools/container_provenance.txt"
test -s "${ROOT}/software/LION-dicom-seg-export/EXPORTER_COMMIT"

TOOL_DIR="${ROOT}/software/dicom-seg-tools"
CODE_ROOT="${ROOT}/software/LION-dicom-seg-export"
DCMQI_SIF="${TOOL_DIR}/dcmqi-v1.5.7.sif"
DICOM3TOOLS_SIF="${TOOL_DIR}/dicom3tools.sif"
sha256sum -c "${ROOT}/software/dicom-seg-tools/SHA256SUMS"
DCMQI_SHA256=$(awk -v path="${DCMQI_SIF}" '$2 == path {print $1}' "${TOOL_DIR}/SHA256SUMS")
DICOM3TOOLS_SHA256=$(awk -v path="${DICOM3TOOLS_SIF}" '$2 == path {print $1}' "${TOOL_DIR}/SHA256SUMS")
EXPORTER_COMMIT=$(tr -d '[:space:]' < "${CODE_ROOT}/EXPORTER_COMMIT")
[[ "${DCMQI_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${DICOM3TOOLS_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${EXPORTER_COMMIT}" =~ ^[0-9a-f]{40}$ ]]

"${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" summarize \
  --expected-json "${QC_ROOT}/dicom_seg_expected.json" \
  --output-json "${QC_ROOT}/dicom_seg_validation.json" \
  --output-csv "${QC_ROOT}/dicom_seg_validation.csv"
"${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" verify-artifacts \
  --report-dir "${QC_ROOT}/reports" \
  --output-root "${OUTPUT_ROOT}" \
  --exporter-commit "${EXPORTER_COMMIT}" \
  --dcmqi-sha256 "${DCMQI_SHA256}" \
  --dicom3tools-sha256 "${DICOM3TOOLS_SHA256}" \
  --output-json "${QC_ROOT}/dicom_seg_artifacts.json"

native_count=$(find "${OUTPUT_ROOT}/native" -maxdepth 1 -type f -name '*.dcm' | wc -l)
suv4_count=$(find "${OUTPUT_ROOT}/suv4" -maxdepth 1 -type f -name '*.dcm' | wc -l)
report_count=$(find "${QC_ROOT}/reports" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ "${native_count}" -eq 133 ]]
[[ "${suv4_count}" -eq 130 ]]
[[ "${report_count}" -eq 266 ]]

"${LION_PY}" - "${QC_ROOT}" <<'PY'
import json
import pathlib
import sys

qc_root = pathlib.Path(sys.argv[1])
summary = json.loads((qc_root / "dicom_seg_validation.json").read_text())
expected = {
    "status": "ok",
    "native": 133,
    "suv4": 130,
    "dicom_seg_objects": 263,
    "unique_series_instance_uids": 263,
    "unique_sop_instance_uids": 263,
    "dciodvfy_errors": 0,
    "dcentvfy_errors": 0,
    "roundtrip_failures": 0,
    "subset_failures": 0,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(f"Invalid cohort summary {key}: {summary.get(key)!r} != {value!r}")
if summary.get("empty_suv4") != [
    "Lung_Dx-A0204",
    "Lung_Dx-A0232",
    "Lung_Dx-A0247",
]:
    raise SystemExit("Unexpected empty SUV>=4 patient list")

slicer = json.loads((qc_root / "viewer/slicer_validation.json").read_text())
if slicer.get("status") != "ok" or slicer.get("segmentation_nodes") != 2:
    raise SystemExit("3D Slicer viewer validation did not pass")
for field in (
    "scene_cleared_before_load",
    "reference_geometry_present",
    "bounds_within_reference",
    "nonempty_binary_labelmaps",
):
    if slicer.get(field) is not True:
        raise SystemExit(f"3D Slicer viewer validation is missing {field}")

ohif = json.loads((qc_root / "viewer/ohif_dicomweb_validation.json").read_text())
if ohif.get("status") != "dicomweb_ok_ui_not_exercised":
    raise SystemExit("OHIF DICOMweb validation evidence is missing or inconsistent")
if ohif.get("instances") != 421 or ohif.get("series") != 3:
    raise SystemExit("OHIF/Orthanc did not expose the complete smoke dataset")
PY

cd "${ROOT}"
shopt -s nullglob
log_files=(
  logs/dicom_seg_tools_*.out
  logs/dicom_seg_tools_*.err
  logs/dicom_seg_export_*.out
  logs/dicom_seg_export_*.err
)
shopt -u nullglob
[[ "${#log_files[@]}" -gt 0 ]]

"${LION_PY}" - "${TEMP_ARCHIVE}" "${log_files[@]}" <<'PY'
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
inputs = [
    pathlib.Path("results/dicom-seg/native"),
    pathlib.Path("results/dicom-seg/suv4"),
    pathlib.Path("results/dicom-seg/qc"),
    pathlib.Path("manifest/fdg_pet_selection.csv"),
    pathlib.Path("qc/segmentation_validation.csv"),
    pathlib.Path("software/dicom-seg-tools/SHA256SUMS"),
    pathlib.Path("software/dicom-seg-tools/container_provenance.txt"),
    pathlib.Path("software/LION-dicom-seg-export/EXPORTER_COMMIT"),
    pathlib.Path("software/LION-dicom-seg-export/README.md"),
    pathlib.Path("software/LION-dicom-seg-export/lionz/dicom_seg.py"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/export_idc_dicom_seg.py"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/validate_slicer_dicom_seg.py"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/lmu/prepare_dicom_seg_tools.sbatch"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/lmu/export_idc_dicom_seg.sbatch"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/lmu/package_dicom_seg.sbatch"),
    pathlib.Path("software/LION-dicom-seg-export/scripts/ohif-smoke"),
    *(pathlib.Path(value) for value in sys.argv[2:]),
]

seen: set[str] = set()
with zipfile.ZipFile(
    archive,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as output:
    for item in inputs:
        if not item.exists():
            raise SystemExit(f"Required archive input does not exist: {item}")
        files = [item] if item.is_file() else sorted(
            candidate for candidate in item.rglob("*") if candidate.is_file()
        )
        for path in files:
            name = path.as_posix()
            if name not in seen:
                output.write(path, arcname=name)
                seen.add(name)

with zipfile.ZipFile(archive) as packaged:
    corrupt = packaged.testzip()
    if corrupt is not None:
        raise SystemExit(f"Archive CRC failure: {corrupt}")
    names = packaged.namelist()

native = sum(
    name.startswith("results/dicom-seg/native/") and name.endswith(".dcm")
    for name in names
)
suv4 = sum(
    name.startswith("results/dicom-seg/suv4/") and name.endswith(".dcm")
    for name in names
)
reports = sum(
    name.startswith("results/dicom-seg/qc/reports/") and name.endswith(".json")
    for name in names
)
if (native, suv4, reports) != (133, 130, 266):
    raise SystemExit(
        f"Unexpected archive counts: native={native}, suv4={suv4}, reports={reports}"
    )

for name in names:
    parts = pathlib.PurePosixPath(name).parts
    if "raw" in parts or "work" in parts:
        raise SystemExit(f"Archive unexpectedly contains intermediate data: {name}")

print(
    f"Archive CRC and inventory OK: {native} native, {suv4} SUV>=4, "
    f"{reports} outcome reports"
)
PY

mv -f "${TEMP_ARCHIVE}" "${ARCHIVE}"
"${LION_PY}" - "${ARCHIVE}" <<'PY'
import hashlib
import pathlib
import sys

archive = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with archive.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
sidecar = archive.with_name(f"{archive.name}.sha256")
sidecar.write_text(f"{digest.hexdigest()}  {archive.name}\n")
print(f"{archive.stat().st_size} bytes  {archive}")
print(sidecar.read_text(), end="")
PY
date -Is
