import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "docker" / "verify_runtime.py"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"


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

    validator.verify_runtime(torch_runtime("2.13.0+cu126", "12.6"))


@pytest.mark.parametrize(
    ("torch_version", "cuda_version"),
    [
        ("2.9.0+cu128", "12.8"),
        ("2.13.0+cu126", "12.8"),
        ("2.13.0", None),
    ],
)
def test_validator_rejects_an_unsupported_cuda_runtime(
    torch_version,
    cuda_version,
):
    validator = load_runtime_validator()

    with pytest.raises(RuntimeError, match="Expected Torch 2.13.0 with CUDA 12.6"):
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


def test_locked_dicom_converter_supports_pydicom_3():
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text())
    dependencies = pyproject["project"]["dependencies"]
    assert "dicom2nifti>=2.6.2" in dependencies

    lockfile = tomllib.loads(UV_LOCK_PATH.read_text())
    locked_package = next(
        package for package in lockfile["package"] if package["name"] == "dicom2nifti"
    )
    assert Version(locked_package["version"]) >= Version("2.6.2")


def test_locked_cuda_wheels_come_from_the_explicit_pytorch_index():
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text())
    lockfile = tomllib.loads(UV_LOCK_PATH.read_text())
    for name, version in (("torch", "2.13.0+cu126"), ("torchvision", "0.28.0+cu126")):
        assert pyproject["tool"]["uv"]["sources"][name]["index"] == "pytorch-cu126"
        packages = [p for p in lockfile["package"] if p["name"] == name and p["version"] == version]
        assert len(packages) == 1
        package = packages[0]
        assert package["source"]["registry"] == "https://download.pytorch.org/whl/cu126"
        for architecture in ("x86_64", "aarch64"):
            assert any(f"manylinux_2_28_{architecture}.whl" in wheel["url"] for wheel in package["wheels"])
