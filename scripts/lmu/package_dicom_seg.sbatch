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

native_count=$(find "${OUTPUT_ROOT}/native" -maxdepth 1 -type f -name '*.dcm' | wc -l)
suv4_count=$(find "${OUTPUT_ROOT}/suv4" -maxdepth 1 -type f -name '*.dcm' | wc -l)
report_count=$(find "${QC_ROOT}/reports" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ "${native_count}" -eq 133 ]]
[[ "${suv4_count}" -eq 130 ]]
[[ "${report_count}" -eq 263 ]]

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

zip -q -r "${TEMP_ARCHIVE}" \
  results/dicom-seg/native \
  results/dicom-seg/suv4 \
  results/dicom-seg/qc \
  manifest/fdg_pet_selection.csv \
  qc/segmentation_validation.csv \
  software/dicom-seg-tools/SHA256SUMS \
  software/dicom-seg-tools/container_provenance.txt \
  software/LION-dicom-seg-export/EXPORTER_COMMIT \
  software/LION-dicom-seg-export/README.md \
  software/LION-dicom-seg-export/lionz/dicom_seg.py \
  software/LION-dicom-seg-export/scripts/export_idc_dicom_seg.py \
  software/LION-dicom-seg-export/scripts/validate_slicer_dicom_seg.py \
  software/LION-dicom-seg-export/scripts/lmu/prepare_dicom_seg_tools.sbatch \
  software/LION-dicom-seg-export/scripts/lmu/export_idc_dicom_seg.sbatch \
  software/LION-dicom-seg-export/scripts/lmu/package_dicom_seg.sbatch \
  software/LION-dicom-seg-export/scripts/ohif-smoke \
  "${log_files[@]}"

unzip -tq "${TEMP_ARCHIVE}"
archive_native=$(unzip -Z1 "${TEMP_ARCHIVE}" | awk '/^results\/dicom-seg\/native\/[^/]+[.]dcm$/ {n++} END {print n+0}')
archive_suv4=$(unzip -Z1 "${TEMP_ARCHIVE}" | awk '/^results\/dicom-seg\/suv4\/[^/]+[.]dcm$/ {n++} END {print n+0}')
archive_reports=$(unzip -Z1 "${TEMP_ARCHIVE}" | awk '/^results\/dicom-seg\/qc\/reports\/[^/]+[.]json$/ {n++} END {print n+0}')
[[ "${archive_native}" -eq 133 ]]
[[ "${archive_suv4}" -eq 130 ]]
[[ "${archive_reports}" -eq 263 ]]

if unzip -Z1 "${TEMP_ARCHIVE}" | grep -Eq '(^|/)(raw|work)(/|$)'; then
  echo "Archive unexpectedly contains raw data or temporary work files" >&2
  exit 1
fi

mv -f "${TEMP_ARCHIVE}" "${ARCHIVE}"
(
  cd "$(dirname "${ARCHIVE}")"
  sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256"
)

ls -lh "${ARCHIVE}" "${ARCHIVE}.sha256"
cat "${ARCHIVE}.sha256"
date -Is
