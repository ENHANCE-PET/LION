#!/usr/bin/env bash
#SBATCH --job-name=lionz-docker-b200-smoke
#SBATCH --account=core-rad-digitx
#SBATCH --partition=jobs-b200
#SBATCH --nodelist=srvcore3
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/docker_candidate_b200_%j.out
#SBATCH --error=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx/logs/docker_candidate_b200_%j.err

set -euo pipefail
umask 002

ROOT=/data2/core-rad-digitx/projects/lionz-lung-pet-ct-dx
PATIENT=Lung_Dx-A0164
CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-lalithshiyam/lionz}"
CANDIDATE_DIGEST="${CANDIDATE_DIGEST:?CANDIDATE_DIGEST is required}"
EXPECTED_VERSION="${EXPECTED_VERSION:?EXPECTED_VERSION is required}"
EXPECTED_REVISION="${EXPECTED_REVISION:?EXPECTED_REVISION is required}"

source "${ROOT}/runtime.env"
test -f "${ROOT}/ALL_OUTPUTS_VALIDATED"
[[ "${CANDIDATE_IMAGE}" =~ ^[a-z0-9._/-]+$ ]]
[[ "${CANDIDATE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ "${EXPECTED_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "${EXPECTED_REVISION}" =~ ^[0-9a-f]{40}$ ]]

DIGEST_HEX="${CANDIDATE_DIGEST#sha256:}"
CANDIDATE_DIR="${ROOT}/software/docker-candidates"
CANDIDATE_SIF="${CANDIDATE_DIR}/lionz-${DIGEST_HEX}.sif"
SMOKE_ROOT="${ROOT}/smoke/docker-candidate-${EXPECTED_VERSION}-${SLURM_JOB_ID}"
mkdir -p \
  "${CANDIDATE_DIR}" \
  "${SMOKE_ROOT}/input/${PATIENT}" \
  "${ROOT}/tmp/singularity-cache"
export SINGULARITY_CACHEDIR="${ROOT}/tmp/singularity-cache"
export SINGULARITY_TMPDIR="${ROOT}/tmp"

if [[ ! -s "${CANDIDATE_SIF}" ]]; then
  temporary="${CANDIDATE_DIR}/.lionz-${DIGEST_HEX}.${SLURM_JOB_ID}.sif"
  singularity pull --disable-cache \
    "${temporary}" \
    "docker://${CANDIDATE_IMAGE}@${CANDIDATE_DIGEST}"
  mv "${temporary}" "${CANDIDATE_SIF}"
fi

cp -a "${ROOT}/raw/${PATIENT}/PT" "${SMOKE_ROOT}/input/${PATIENT}/"
export LIONZ_TELEMETRY=0
export MPLCONFIGDIR="${ROOT}/mpl-cache"

COMMON_ARGS=(
  --nv
  --bind "${SMOKE_ROOT}/input:/shared"
  --bind "${LION_MODELS}:/usr/local/models"
  --bind "${ROOT}/tmp:/tmp"
  "${CANDIDATE_SIF}"
)

hostname
date -Is
nvidia-smi
singularity exec "${COMMON_ARGS[@]}" \
  env EXPECTED_VERSION="${EXPECTED_VERSION}" EXPECTED_REVISION="${EXPECTED_REVISION}" \
  python -c 'import importlib.metadata as metadata, os, torch; assert metadata.version("lionz") == os.environ["EXPECTED_VERSION"]; assert torch.cuda.get_device_name(0).startswith("NVIDIA B200"); assert torch.arange(8, device="cuda").sum().item() == 28; print("lionz", metadata.version("lionz"), "torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))'
singularity exec "${COMMON_ARGS[@]}" lionz -d /shared -m fdg

SEGMENTATION=$(find "${SMOKE_ROOT}/input/${PATIENT}" -path '*/lionz-*/segmentations/*.nii.gz' -type f -print -quit)
test -n "${SEGMENTATION}"
test -s "${SEGMENTATION}"
REFERENCE=$("${LION_PY}" - "${ROOT}/qc/segmentation_validation.csv" "${PATIENT}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    matches = [row for row in csv.DictReader(stream) if row["PatientID"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"Expected one QC row, found {len(matches)}")
print(matches[0]["native_path"])
PY
)

"${LION_PY}" - "${SEGMENTATION}" "${REFERENCE}" <<'PY'
import sys

import numpy as np
import SimpleITK as sitk

actual = sitk.ReadImage(sys.argv[1])
expected = sitk.ReadImage(sys.argv[2])
assert actual.GetSize() == expected.GetSize()
assert np.allclose(actual.GetSpacing(), expected.GetSpacing(), rtol=0, atol=1e-6)
assert np.allclose(actual.GetOrigin(), expected.GetOrigin(), rtol=0, atol=1e-6)
assert np.allclose(actual.GetDirection(), expected.GetDirection(), rtol=0, atol=1e-6)
actual_array = sitk.GetArrayFromImage(actual)
expected_array = sitk.GetArrayFromImage(expected)
assert np.array_equal(actual_array, expected_array)
assert int(np.count_nonzero(actual_array)) > 0
print(f"Real-data candidate output exactly matches {sys.argv[2]}")
PY

{
  printf 'candidate_image=%s\n' "${CANDIDATE_IMAGE}"
  printf 'candidate_digest=%s\n' "${CANDIDATE_DIGEST}"
  printf 'expected_version=%s\n' "${EXPECTED_VERSION}"
  printf 'expected_revision=%s\n' "${EXPECTED_REVISION}"
  printf 'patient=%s\n' "${PATIENT}"
  printf 'segmentation=%s\n' "${SEGMENTATION}"
  printf 'validated_at=%s\n' "$(date -Is)"
} > "${SMOKE_ROOT}/VALIDATED"

cat "${SMOKE_ROOT}/VALIDATED"
