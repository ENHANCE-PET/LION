#!/usr/bin/env bash
#SBATCH --job-name=lionz-idc-dicom-seg
#SBATCH --account=core-rad-digitx
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_export_%j.out
#SBATCH --error=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_export_%j.err

set -euo pipefail
umask 002

ROOT=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx
CODE_ROOT="${CODE_ROOT:-${ROOT}/software/LION-dicom-seg-export}"
MODE="${MODE:-full}"
WORKERS="${WORKERS:-8}"
OUTPUT_ROOT="${ROOT}/results/dicom-seg"
TOOL_DIR="${ROOT}/software/dicom-seg-tools"
DCMQI_SIF="${TOOL_DIR}/dcmqi-v1.5.7.sif"
DICOM3TOOLS_SIF="${TOOL_DIR}/dicom3tools.sif"
LION_SHA=39a2139b54895ab37e10affe21a4d38b785413ad

source "${ROOT}/runtime.env"
test -f "${ROOT}/ALL_OUTPUTS_VALIDATED"
test -f "${TOOL_DIR}/TOOLS_READY"
test -s "${DCMQI_SIF}"
test -s "${DICOM3TOOLS_SIF}"
test -s "${TOOL_DIR}/SHA256SUMS"
test -f "${CODE_ROOT}/scripts/export_idc_dicom_seg.py"
test -f "${CODE_ROOT}/lionz/dicom_seg.py"
[[ "${MODE}" == "smoke" || "${MODE}" == "full" ]]
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]

export PYTHONPATH="${CODE_ROOT}"
export TMPDIR="${ROOT}/tmp"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
mkdir -p "${OUTPUT_ROOT}" "${ROOT}/logs" "${ROOT}/tmp"
rm -f "${OUTPUT_ROOT}/qc/COHORT_VALIDATED"
rm -f "${OUTPUT_ROOT}/qc/SMOKE_VALIDATED"
rm -f "${OUTPUT_ROOT}/qc/viewer/slicer_validation.json"
rm -f "${OUTPUT_ROOT}/qc/viewer/ohif_dicomweb_validation.json"

sha256sum -c "${TOOL_DIR}/SHA256SUMS"
DCMQI_SHA256=$(awk -v path="${DCMQI_SIF}" '$2 == path {print $1}' "${TOOL_DIR}/SHA256SUMS")
DICOM3TOOLS_SHA256=$(awk -v path="${DICOM3TOOLS_SIF}" '$2 == path {print $1}' "${TOOL_DIR}/SHA256SUMS")
EXPORTER_COMMIT=$(tr -d '[:space:]' < "${CODE_ROOT}/EXPORTER_COMMIT")
[[ "${DCMQI_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${DICOM3TOOLS_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${EXPORTER_COMMIT}" =~ ^[0-9a-f]{40}$ ]]

"${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" prepare \
  --selection-csv "${ROOT}/manifest/fdg_pet_selection.csv" \
  --qc-csv "${ROOT}/qc/segmentation_validation.csv" \
  --project-dir "${ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-commit "${LION_SHA}" \
  --exporter-commit "${EXPORTER_COMMIT}" \
  --dcmqi-sha256 "${DCMQI_SHA256}" \
  --dicom3tools-sha256 "${DICOM3TOOLS_SHA256}"

TASKS_FILE="${OUTPUT_ROOT}/qc/dicom_seg_tasks.jsonl"
EXPECTED_JSON="${OUTPUT_ROOT}/qc/dicom_seg_expected.json"
TASK_COUNT=$(wc -l < "${TASKS_FILE}")
[[ "${TASK_COUNT}" -eq 263 ]]

run_one() {
  "${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" convert-one \
    --tasks-file "${TASKS_FILE}" \
    "$@" \
    --dcmqi-sif "${DCMQI_SIF}" \
    --dicom3tools-sif "${DICOM3TOOLS_SIF}" \
    --singularity /usr/bin/singularity
}

hostname
date -Is
run_one --patient-id Lung_Dx-A0164 --variant native
run_one --patient-id Lung_Dx-A0164 --variant suv4
touch "${OUTPUT_ROOT}/qc/SMOKE_VALIDATED"

if [[ "${MODE}" == "smoke" ]]; then
  date -Is
  exit 0
fi

export LION_PY CODE_ROOT TASKS_FILE DCMQI_SIF DICOM3TOOLS_SIF
seq 1 "${TASK_COUNT}" | xargs -P "${WORKERS}" -I TASK_INDEX bash -c '
  "${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" convert-one \
    --tasks-file "${TASKS_FILE}" \
    --task-index TASK_INDEX \
    --dcmqi-sif "${DCMQI_SIF}" \
    --dicom3tools-sif "${DICOM3TOOLS_SIF}" \
    --singularity /usr/bin/singularity
'

"${LION_PY}" "${CODE_ROOT}/scripts/export_idc_dicom_seg.py" summarize \
  --expected-json "${EXPECTED_JSON}" \
  --output-json "${OUTPUT_ROOT}/qc/dicom_seg_validation.json" \
  --output-csv "${OUTPUT_ROOT}/qc/dicom_seg_validation.csv"

touch "${OUTPUT_ROOT}/qc/COHORT_VALIDATED"
date -Is
