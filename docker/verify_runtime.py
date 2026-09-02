"""Verify the CUDA runtime contract baked into the LION Docker image."""

EXPECTED_TORCH_VERSION = "2.6.0"
EXPECTED_CUDA_VERSION = "12.4"


def verify_runtime(torch_module) -> None:
    """Raise when Torch does not match the supported Docker GPU runtime."""
    torch_version = str(torch_module.__version__).split("+", 1)[0]
    cuda_version = torch_module.version.cuda

    if (
        torch_version != EXPECTED_TORCH_VERSION
        or cuda_version != EXPECTED_CUDA_VERSION
    ):
        raise RuntimeError(
            f"Expected Torch {EXPECTED_TORCH_VERSION} with CUDA "
            f"{EXPECTED_CUDA_VERSION}; found Torch {torch_module.__version__} "
            f"with CUDA {cuda_version}."
        )


def main() -> None:
    import torch

    verify_runtime(torch)
    print(f"Verified Torch {torch.__version__} with CUDA {torch.version.cuda}")


if __name__ == "__main__":
    main()
