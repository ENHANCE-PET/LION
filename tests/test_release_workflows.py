from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"


def workflow_text(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def test_candidate_tag_is_derived_and_cannot_be_supplied_by_a_dispatcher():
    workflow = workflow_text("docker-candidate.yml")

    dispatch_inputs = workflow.split("concurrency:", 1)[0]
    assert "candidate_tag:" not in dispatch_inputs
    assert 'candidate_tag="${LIONZ_VERSION}-candidate-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"' in workflow
    assert 'expected_tag="lionz-v.${LIONZ_VERSION}"' in workflow


def test_promotion_is_bound_to_a_candidate_tag_and_protected_environment():
    workflow = workflow_text("docker-promote.yml")

    assert "candidate_tag:" in workflow
    assert 'expected_prefix="${LIONZ_VERSION}-candidate-"' in workflow
    assert "environment: dockerhub-production" in workflow
    assert 'org.opencontainers.image.revision' in workflow


def test_release_runtime_uses_a_torch_supported_python_version():
    workflow = workflow_text("release-publish.yml")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert 'python-version: "3.12"' in workflow
    assert 'requires-python = ">=3.10,<3.14"' in pyproject
