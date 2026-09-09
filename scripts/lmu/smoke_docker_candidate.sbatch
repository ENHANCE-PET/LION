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
SMOKE_ROOT="${ROOT}/smoke/docker-candidate-${EXPECTED_VERSION}-${SLURM_JOB_ID}"
CANDIDATE_SIF="${SMOKE_ROOT}/lionz-${DIGEST_HEX}.sif"
mkdir -p \
  "${SMOKE_ROOT}/input/${PATIENT}" \
  "${SMOKE_ROOT}/validation" \
  "${SMOKE_ROOT}/tmp" \
  "${SMOKE_ROOT}/singularity-cache"
export SINGULARITY_CACHEDIR="${SMOKE_ROOT}/singularity-cache"
export SINGULARITY_TMPDIR="${SMOKE_ROOT}/tmp"

PULLED_REFERENCE="docker://${CANDIDATE_IMAGE}@${CANDIDATE_DIGEST}"
singularity pull --disable-cache "${CANDIDATE_SIF}" "${PULLED_REFERENCE}"
test -s "${CANDIDATE_SIF}"
SIF_SHA256=$(sha256sum "${CANDIDATE_SIF}" | awk '{print $1}')
INSPECT_JSON="${SMOKE_ROOT}/validation/singularity_inspect.json"
singularity inspect --json "${CANDIDATE_SIF}" > "${INSPECT_JSON}"

IFS=$'\t' read -r OBSERVED_VERSION OBSERVED_REVISION OBSERVED_SOURCE BUILD_FROM < <(
  "${LION_PY}" - "${INSPECT_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
labels = payload["data"]["attributes"]["labels"]
build_from = [
    str(value)
    for key, value in labels.items()
    if key.endswith(".deffile.from")
]
if len(build_from) != 1:
    raise SystemExit(f"Expected one Singularity deffile.from label: {build_from}")
print(
    labels.get("org.opencontainers.image.version", ""),
    labels.get("org.opencontainers.image.revision", ""),
    labels.get("org.opencontainers.image.source", ""),
    build_from[0],
    sep="\t",
)
PY
)
[[ "${BUILD_FROM}" == *"@${CANDIDATE_DIGEST}" ]]
OBSERVED_DIGEST="${BUILD_FROM##*@}"
[[ "${OBSERVED_DIGEST}" == "${CANDIDATE_DIGEST}" ]]
[[ "${OBSERVED_VERSION}" == "${EXPECTED_VERSION}" ]]
[[ "${OBSERVED_REVISION}" == "${EXPECTED_REVISION}" ]]
[[ "${OBSERVED_SOURCE}" == "https://github.com/ENHANCE-PET/LION" ]]

cp -a "${ROOT}/raw/${PATIENT}/PT" "${SMOKE_ROOT}/input/${PATIENT}/"

COMMON_ARGS=(
  --nv
  --containall
  --cleanenv
  --no-home
  --pwd /shared
  --bind "${SMOKE_ROOT}/input:/shared"
  --bind "${LION_MODELS}:/usr/local/models:ro"
  --bind "${SMOKE_ROOT}/tmp:/tmp"
  --env LIONZ_TELEMETRY=0
  --env MPLCONFIGDIR=/tmp/mpl-cache
  "${CANDIDATE_SIF}"
)

hostname
date -Is
nvidia-smi
singularity exec "${COMMON_ARGS[@]}" \
  python - "${EXPECTED_VERSION}" > "${SMOKE_ROOT}/validation/runtime.json" <<'PY'
import importlib.metadata as metadata
import json
import sys

import torch

version = metadata.version("lionz")
gpu = torch.cuda.get_device_name(0)
assert version == sys.argv[1]
assert gpu.startswith("NVIDIA B200")
assert torch.arange(8, device="cuda").sum().item() == 28
print(json.dumps({
    "lionz_version": version,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "gpu": gpu,
}, sort_keys=True))
PY
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

"${LION_PY}" - "${SEGMENTATION}" "${REFERENCE}" \
  > "${SMOKE_ROOT}/validation/output_comparison.json" <<'PY'
import json
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
print(json.dumps({
    "exact_match": True,
    "nonzero_voxels": int(np.count_nonzero(actual_array)),
    "reference_path": sys.argv[2],
}, sort_keys=True))
PY

SEGMENTATION_SHA256=$(sha256sum "${SEGMENTATION}" | awk '{print $1}')
REFERENCE_SHA256=$(sha256sum "${REFERENCE}" | awk '{print $1}')
HOSTNAME=$(hostname)
VALIDATED_AT=$(date -Is)
export CANDIDATE_IMAGE CANDIDATE_DIGEST EXPECTED_VERSION EXPECTED_REVISION
export OBSERVED_DIGEST OBSERVED_VERSION OBSERVED_REVISION OBSERVED_SOURCE
export PULLED_REFERENCE BUILD_FROM SIF_SHA256 PATIENT SEGMENTATION
export SEGMENTATION_SHA256 REFERENCE REFERENCE_SHA256 HOSTNAME VALIDATED_AT
export SLURM_JOB_ID SMOKE_ROOT
"${LION_PY}" - > "${SMOKE_ROOT}/VALIDATED.json" <<'PY'
import json
import os
from pathlib import Path

runtime = json.loads(Path(os.environ["SMOKE_ROOT"], "validation/runtime.json").read_text())
comparison = json.loads(
    Path(os.environ["SMOKE_ROOT"], "validation/output_comparison.json").read_text()
)
report = {
    "status": "validated",
    "slurm_job_id": int(os.environ["SLURM_JOB_ID"]),
    "hostname": os.environ["HOSTNAME"],
    "candidate_image": os.environ["CANDIDATE_IMAGE"],
    "requested_digest": os.environ["CANDIDATE_DIGEST"],
    "observed_digest": os.environ["OBSERVED_DIGEST"],
    "expected_version": os.environ["EXPECTED_VERSION"],
    "observed_version": os.environ["OBSERVED_VERSION"],
    "expected_revision": os.environ["EXPECTED_REVISION"],
    "observed_revision": os.environ["OBSERVED_REVISION"],
    "observed_source": os.environ["OBSERVED_SOURCE"],
    "pulled_reference": os.environ["PULLED_REFERENCE"],
    "singularity_build_from": os.environ["BUILD_FROM"],
    "sif_sha256": os.environ["SIF_SHA256"],
    "patient": os.environ["PATIENT"],
    "segmentation": os.environ["SEGMENTATION"],
    "segmentation_sha256": os.environ["SEGMENTATION_SHA256"],
    "reference": os.environ["REFERENCE"],
    "reference_sha256": os.environ["REFERENCE_SHA256"],
    "output_exact_match": comparison["exact_match"],
    "nonzero_voxels": comparison["nonzero_voxels"],
    "runtime": runtime,
    "models_read_only": True,
    "contained_execution": True,
    "validated_at": os.environ["VALIDATED_AT"],
}
print(json.dumps(report, indent=2, sort_keys=True))
PY
sha256sum "${SMOKE_ROOT}/VALIDATED.json" > "${SMOKE_ROOT}/VALIDATED.json.sha256"
cat "${SMOKE_ROOT}/VALIDATED.json"
cat "${SMOKE_ROOT}/VALIDATED.json.sha256"
