import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "docker" / "verify_runtime.py"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"


def load_runtime_validator():
    assert VALIDATOR_PATH.is_file(), "Docker runtime validator is missing"
    spec = importlib.util.spec_from_file_location("verify_runtime", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_runtime(torch_version: str, cuda_version: str | None):
    return SimpleNamespace(
        __version__=torch_version,
        version=SimpleNamespace(cuda=cuda_version),
    )


def test_validator_accepts_the_supported_cuda_runtime():
    validator = load_runtime_validator()

    validator.verify_runtime(torch_runtime("2.6.0+cu124", "12.4"))


@pytest.mark.parametrize(
    ("torch_version", "cuda_version"),
    [
        ("2.9.0+cu128", "12.8"),
        ("2.6.0+cu124", "12.8"),
        ("2.6.0", None),
    ],
)
def test_validator_rejects_an_unsupported_cuda_runtime(
    torch_version,
    cuda_version,
):
    validator = load_runtime_validator()

    with pytest.raises(RuntimeError, match="Expected Torch 2.6.0 with CUDA 12.4"):
        validator.verify_runtime(torch_runtime(torch_version, cuda_version))


def test_docker_image_installs_custom_trainer_before_runtime_verification():
    dockerfile = DOCKERFILE_PATH.read_text()
    installer = "add_custom_trainers_to_local_nnunetv2"
    installed_import = (
        "nnunetv2.training.nnUNetTrainer.variants.LION_custom_trainers"
    )
    runtime_verification = "python /tmp/verify_runtime.py"

    assert installer in dockerfile
    assert installed_import in dockerfile
    assert dockerfile.index(installer) < dockerfile.index(runtime_verification)
