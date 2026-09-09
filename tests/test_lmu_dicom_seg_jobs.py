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
    assert 'sha256sum -c "${TOOL_DIR}/SHA256SUMS"' in script


def test_packaging_rechecks_current_artifacts_and_tool_hashes():
    script = _text("scripts/lmu/package_dicom_seg.sbatch")

    assert 'sha256sum -c "${ROOT}/software/dicom-seg-tools/SHA256SUMS"' in script
    assert "verify-artifacts" in script
    assert "summarize" in script


def test_ohif_smoke_endpoint_is_loopback_only():
    compose = _text("scripts/ohif-smoke/docker-compose.yml")

    assert '"127.0.0.1:3001:80"' in compose
    assert '"8042:8042"' not in compose
