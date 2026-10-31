# LION 1.1.0

This security release updates vulnerable dependencies and restricts untrusted
model archives and MCP organization plans. It retains the FDG/PSMA model URLs,
public `lion` function signature, and CLI inference options. Python 3.11 or newer
is required; the supported range is 3.11–3.13. The minor version signals this
runtime migration from 1.0.7.

## Security changes

- Updates Torch to 2.13.0, torchvision to 0.28.0, Pillow to 13.0.0, pydicom to
  3.0.2, and vulnerable networking, archive, and file-lock dependencies.
- Verifies model downloads against pinned SHA-256 digests before extraction.
  Archives cannot write outside their expected model directory. Failed model
  upgrades preserve the previous cache; installation failures roll back.
- Constrains MCP plan source/destination paths and prevents output overwrites.
- Removes the dormant Windows helper's broad executable write permission grant.
- Transfers release tags through an environment variable before Bash validation,
  preventing release/dispatch tag text from executing shell commands.

## Upgrade and compatibility

Docker and locked Linux/Windows installations use the official patched
[CUDA 12.6 Torch build](https://pytorch.org/get-started/previous-versions/#v2130),
available for Linux x86_64/aarch64 and Windows x86_64. Docker publication currently
builds only Linux amd64. CUDA 12.6's Linux toolkit-driver baseline is 560.28.03;
NVIDIA documents older-driver minor compatibility with feature/PTX restrictions.
See the [NVIDIA driver table](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html).

Plain `pip install lionz` uses PyPI's Linux Torch build, which targets CUDA 13.0
and requires a compatible 580-series driver. The README provides the two-step
installation command to select CUDA 12.6 explicitly. macOS uses the PyPI build
with CPU/MPS support. No model weights or clinical fixtures are included.

Legacy model caches with only a URL marker must be refreshed from a verified
archive. Run once with writable cache storage before using a read-only model
mount. A failed refresh preserves existing weights and reports an error.

## Validation and remaining gate

The complete candidate suite passed 74 tests, including explicit CUDA-index/wheel
selection and safe Apple archive metadata handling. The built 1.1.0 wheel was
installed and tested outside the source checkout: its console help, public API
signature, invalid-input handling, version metadata, nnUNet CPU predictor
initialization, and a synthetic Torch Conv3d calculation all passed. Version 1.1.0
wheel and source distribution built successfully and passed `twine check`.
The test environment uses the locked macOS inference stack; build-tool installation
upgraded the auxiliary `rich`/`packaging` packages to 15.0.0/26.3. The earlier
70-test security suite also passed with the exact locked 14.2.0/25.0 versions.

The `Test Locked Runtime` pull-request workflow runs the complete locked suite
on Python 3.12/Linux without publication credentials. Its result must pass before
merging; local YAML/Bash syntax and real Bash release-tag regressions passed.

These checks do not establish unchanged segmentation output on a GPU. The Docker
candidate must pass the existing LMU B200 exact-output test and produce its
attestation before stable Docker promotion. No GPU inference or patient/model
download was performed during release preparation.

## Publishing sequence

1. Merge the reviewed branch and tag the resulting commit `lionz-v.1.1.0` without
   changing existing tags. Publish the GitHub release with these notes. The
   `release-publish.yml` workflow checks the tag/version, runs the locked test
   suite, builds and validates wheel/sdist, then publishes to PyPI using
   `PYPI_LIONZ_TOKEN` (fallback `PYPI_API_TOKEN`). It verifies exact uploaded
   artifact hashes. Its manual dispatch accepts the same immutable tag.
2. Dispatch `docker-candidate.yml` with `version=1.1.0`. It checks out that tag,
   runs the locked suite, and builds a Linux amd64 image using the pinned Python
   base. The Docker build imports the custom nnUNet trainer and verifies Torch
   2.13.0 / CUDA 12.6. `DOCKER_USERNAME` and `DOCKER_PASSWORD` are required. The
   workflow pushes only a derived candidate tag and records its exact digest.
3. Test that exact digest with `scripts/lmu/smoke_docker_candidate.sbatch`, using
   a verified read-only model cache and the established reference output. Attach
   `lionz-1.1.0-b200-smoke.json` to the GitHub release. No attestation may be
   synthesized or reused for another digest/revision.
4. Dispatch `docker-promote.yml` with version, candidate tag, and tested digest.
   The `dockerhub-production` environment may require a configured reviewer.
   The workflow verifies the B200 attestation, PyPI version, image labels,
   architecture, and runtime, then promotes the same digest to `1.1.0`, `v1.1.0`,
   and `latest` without rebuilding. GPU validation and any environment approval
   remain required before this step can succeed.

The repository's `lion-mcp` subproject is fixed and independently locked; the
root PyPI workflow publishes `lionz` only and does not publish a separate MCP
package.
