from pathlib import Path


REPOSITORY = Path(__file__).parents[1]


def _text(relative_path: str) -> str:
    return (REPOSITORY / relative_path).read_text()


def test_tool_bootstrap_uses_immutable_oci_digests():
    script = _text("scripts/lmu/prepare_dicom_seg_tools.sbatch")

    assert "docker://qiicr/dcmqi@sha256:" in script
    assert "docker://qiicr/dicom3tools@sha256:" in script
    assert "qiicr/dicom3tools:latest" not in script
    assert 'rm -f "${TOOL_DIR}/TOOLS_READY"' in script
    assert 'if [[ ! -s "${DCMQI_SIF}" ]]' not in script


def test_export_invalidates_markers_and_verifies_sif_hashes():
    script = _text("scripts/lmu/export_idc_dicom_seg.sbatch")

    assert 'rm -f "${OUTPUT_ROOT}/qc/COHORT_VALIDATED"' in script
    assert 'rm -f "${OUTPUT_ROOT}/qc/SMOKE_VALIDATED"' in script
    assert 'rm -f "${OUTPUT_ROOT}/qc/viewer/slicer_validation.json"' in script
    assert 'rm -f "${OUTPUT_ROOT}/qc/viewer/ohif_dicomweb_validation.json"' in script
    assert 'sha256sum -c "${TOOL_DIR}/SHA256SUMS"' in script


def test_packaging_rechecks_current_artifacts_and_tool_hashes():
    script = _text("scripts/lmu/package_dicom_seg.sbatch")

    assert 'sha256sum -c "${ROOT}/software/dicom-seg-tools/SHA256SUMS"' in script
    assert "verify-artifacts" in script
    assert "summarize" in script
    assert 'viewer.get("seg_objects"' in script


def test_ohif_smoke_endpoint_is_loopback_only():
    compose = _text("scripts/ohif-smoke/docker-compose.yml")

    assert '"127.0.0.1:3001:80"' in compose
    assert '"8042:8042"' not in compose


def test_viewer_validators_bind_evidence_to_seg_hashes_and_commit():
    slicer_script = _text("scripts/validate_slicer_dicom_seg.py")
    ohif_script = _text("scripts/validate_ohif_dicomweb.py")

    for script in (slicer_script, ohif_script):
        assert "exporter_commit" in script
        assert "seg_objects" in script
        assert "sha256" in script
        assert "sop_instance_uid" in script
        assert "series_instance_uid" in script


def test_docker_candidate_smoke_uses_digest_b200_and_real_data():
    script = _text("scripts/lmu/smoke_docker_candidate.sbatch")

    assert "#SBATCH --partition=jobs-b200" in script
    assert "#SBATCH --gres=gpu:b200:1" in script
    assert '[[ "${CANDIDATE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]' in script
    assert "Lung_Dx-A0164" in script
    assert 'raw/${PATIENT}/PT' in script
    assert "np.array_equal" in script
