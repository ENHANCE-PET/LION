#!/usr/bin/env bash
#SBATCH --job-name=lionz-dicom-seg-tools
#SBATCH --account=core-rad-digitx
#SBATCH --partition=jobs-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_tools_%j.out
#SBATCH --error=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/dicom_seg_tools_%j.err

set -euo pipefail
umask 002

ROOT=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx
TOOL_DIR="${ROOT}/software/dicom-seg-tools"
DCMQI_SIF="${TOOL_DIR}/dcmqi-v1.5.7.sif"
DICOM3TOOLS_SIF="${TOOL_DIR}/dicom3tools.sif"
DCMQI_IMAGE=docker://qiicr/dcmqi@sha256:7ab93b32ecb1c304813ec85bc2ffcbbc834f406b797b48ca2b46df94838e906e
DICOM3TOOLS_IMAGE=docker://qiicr/dicom3tools@sha256:f3bd33512bd48d4fa85b7b794aa98475e37965e95356c8ece1c68ff74264f10f

mkdir -p "${TOOL_DIR}" "${ROOT}/logs" "${ROOT}/tmp/singularity-cache"
export SINGULARITY_CACHEDIR="${ROOT}/tmp/singularity-cache"
export SINGULARITY_TMPDIR="${ROOT}/tmp"

rm -f "${TOOL_DIR}/TOOLS_READY"
DCMQI_TEMPORARY="${TOOL_DIR}/.dcmqi-v1.5.7.${SLURM_JOB_ID}.sif"
DICOM3TOOLS_TEMPORARY="${TOOL_DIR}/.dicom3tools.${SLURM_JOB_ID}.sif"
cleanup() {
  rm -f "${DCMQI_TEMPORARY}" "${DICOM3TOOLS_TEMPORARY}"
}
trap cleanup EXIT

singularity pull --disable-cache "${DCMQI_TEMPORARY}" "${DCMQI_IMAGE}"
singularity pull --disable-cache "${DICOM3TOOLS_TEMPORARY}" "${DICOM3TOOLS_IMAGE}"
mv "${DCMQI_TEMPORARY}" "${DCMQI_SIF}"
mv "${DICOM3TOOLS_TEMPORARY}" "${DICOM3TOOLS_SIF}"

sha256sum "${DCMQI_SIF}" "${DICOM3TOOLS_SIF}" > "${TOOL_DIR}/SHA256SUMS"
sha256sum -c "${TOOL_DIR}/SHA256SUMS"
{
  printf 'dcmqi_image=%s\n' "${DCMQI_IMAGE}"
  printf 'dicom3tools_image=%s\n' "${DICOM3TOOLS_IMAGE}"
  printf 'created_at=%s\n' "$(date -Is)"
  singularity inspect "${DCMQI_SIF}"
  singularity inspect "${DICOM3TOOLS_SIF}"
} > "${TOOL_DIR}/container_provenance.txt"

singularity exec "${DCMQI_SIF}" sh -c 'command -v itkimage2segimage; command -v segimage2itkimage'
singularity exec "${DICOM3TOOLS_SIF}" test -x /usr/src/dicom3tools/bin/1.4.4.0.x8664/dciodvfy
singularity exec "${DICOM3TOOLS_SIF}" test -x /usr/src/dicom3tools/bin/1.4.4.0.x8664/dcentvfy
touch "${TOOL_DIR}/TOOLS_READY"
date -Is
