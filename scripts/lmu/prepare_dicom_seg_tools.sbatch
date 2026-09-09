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

mkdir -p "${TOOL_DIR}" "${ROOT}/logs" "${ROOT}/tmp/singularity-cache"
export SINGULARITY_CACHEDIR="${ROOT}/tmp/singularity-cache"
export SINGULARITY_TMPDIR="${ROOT}/tmp"

if [[ ! -s "${DCMQI_SIF}" ]]; then
  temporary="${TOOL_DIR}/.dcmqi-v1.5.7.${SLURM_JOB_ID}.sif"
  singularity pull --disable-cache "${temporary}" docker://qiicr/dcmqi:v1.5.7
  mv "${temporary}" "${DCMQI_SIF}"
fi
if [[ ! -s "${DICOM3TOOLS_SIF}" ]]; then
  temporary="${TOOL_DIR}/.dicom3tools.${SLURM_JOB_ID}.sif"
  singularity pull --disable-cache "${temporary}" docker://qiicr/dicom3tools:latest
  mv "${temporary}" "${DICOM3TOOLS_SIF}"
fi

sha256sum "${DCMQI_SIF}" "${DICOM3TOOLS_SIF}" > "${TOOL_DIR}/SHA256SUMS"
{
  printf 'dcmqi_image=docker://qiicr/dcmqi:v1.5.7\n'
  printf 'dicom3tools_image=docker://qiicr/dicom3tools:latest\n'
  printf 'created_at=%s\n' "$(date -Is)"
  singularity inspect "${DCMQI_SIF}"
  singularity inspect "${DICOM3TOOLS_SIF}"
} > "${TOOL_DIR}/container_provenance.txt"

singularity exec "${DCMQI_SIF}" sh -c 'command -v itkimage2segimage; command -v segimage2itkimage'
singularity exec "${DICOM3TOOLS_SIF}" test -x /usr/src/dicom3tools/bin/1.4.4.0.x8664/dciodvfy
singularity exec "${DICOM3TOOLS_SIF}" test -x /usr/src/dicom3tools/bin/1.4.4.0.x8664/dcentvfy
touch "${TOOL_DIR}/TOOLS_READY"
date -Is
