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
    assert 'lionz-${LIONZ_VERSION}-b200-smoke.json' in workflow
    assert 'report["observed_digest"]' in workflow
    assert 'report["observed_revision"]' in workflow


def test_release_runtime_uses_a_torch_supported_python_version():
    workflow = workflow_text("release-publish.yml")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()

    assert 'python-version: "3.12"' in workflow
    assert 'requires-python = ">=3.11,<3.14"' in pyproject


def release_metadata_script(event: str, tag: str) -> str:
    import yaml

    workflow = yaml.safe_load(workflow_text("release-publish.yml"))
    script = workflow["jobs"]["release-meta"]["steps"][0]["run"]
    # Reproduce GitHub expression substitution for the old unsafe workflow too.
    replacements = {
        "github.event_name": event,
        "github.event.release.tag_name": tag if event == "release" else "",
        "github.event.inputs.tag": tag if event == "workflow_dispatch" else "",
    }
    for expression, value in replacements.items():
        script = script.replace("${{ " + expression + " }}", value)
    return script


def test_release_and_dispatch_payloads_cannot_execute_shell(tmp_path):
    import os
    import shlex
    import subprocess

    for event in ("release", "workflow_dispatch"):
        marker = tmp_path / f"{event}-executed"
        tag = f"$(touch {shlex.quote(str(marker))})"
        environment = {
            **os.environ,
            "RELEASE_TAG": tag,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        }
        result = subprocess.run(
            ["bash", "-c", release_metadata_script(event, tag)],
            env=environment, capture_output=True, text=True, timeout=5,
        )
        assert result.returncode != 0
        assert not marker.exists(), result.stderr


def test_valid_release_and_dispatch_tags_generate_expected_metadata(tmp_path):
    import os
    import subprocess

    tag = "lionz-v.1.1.0"
    for event in ("release", "workflow_dispatch"):
        output = tmp_path / event
        result = subprocess.run(
            ["bash", "-c", release_metadata_script(event, tag)],
            env={**os.environ, "RELEASE_TAG": tag, "GITHUB_OUTPUT": str(output)},
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert output.read_text() == "tag=lionz-v.1.1.0\nversion=1.1.0\n"
