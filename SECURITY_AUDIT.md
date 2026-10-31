# Security audit — 2026-10-31

Scope: LION runtime, release configuration, model downloads, and the standalone
`lion-mcp` package. No clinical documents, model weights, or patient fixtures were
downloaded or added. The security branch is prepared for the separately authorized 1.1.0 release.

## Findings and repairs

- GitHub reported **100 open dependency alerts covering 92 distinct GHSA IDs**.
  The response had no next-page link. The updated root lock excludes every
  supplied affected version range. Security floors cover Pillow 13.0.0,
  Torch 2.13.0 / torchvision 0.28.0, pydicom 3.0.2, AnyIO 4.14.2,
  certifi 2026.10.18, filelock 4.2.1, fsspec 2026.6.0, msgpack 1.2.3,
  setuptools 84.1.0, Tornado 6.6, and urllib3 2.8.0.
- MCP now requires the patched 1.30.x SDK line and has its own lockfile.
  Organization plans must keep sources within `source_dir`, use single-component
  destination names, reject escaping symlinks, and create output files
  exclusively rather than overwrite existing data. Synthetic tests reproduce
  unauthorized source copying/moving, traversal, symlink escape, and overwrite.
- Model archives are streamed into private temporary storage, checked against
  SHA-256 digests from the existing GitHub release assets, and validated before
  extraction. Members must remain under the expected model directory and cannot
  be symlinks or contain traversal/Windows path syntax. HTTP requests have
  connection/read timeouts. Existing caches remain intact until replacement
  validation finishes; installation backs up the previous cache and rolls back
  on failure. A backup is retained if rollback itself fails.
- Release and manual-dispatch tags are transferred through an environment
  variable before Bash validation. Real Bash regressions reproduce and prevent
  pre-validation command substitution, and verify valid tags still produce the
  expected version metadata. New pull-request CI runs the locked full suite on
  Python 3.12/Linux with read-only permissions and no publication secrets.
- The dormant Windows permission helper no longer grants `Everyone` full control
  over an executable. No active executable download/cache path calls this helper;
  pre-existing filesystem ACLs are not repaired by this change.

## Compatibility and validation

Patched Pillow and Tornado require **Python 3.11+**. The Docker base now uses a
verified, digest-pinned Python 3.12 image. Official Torch 2.13.0 and torchvision
0.28.0 CUDA 12.6 wheels are selected by an explicit PyTorch index for locked
Linux/Windows installations, including Linux x86_64 and aarch64. Docker runtime
checks and promotion expect CUDA 12.6. NVIDIA's CUDA 12.6 GA driver baseline is
560.28.03 on Linux; older drivers have limited minor-version compatibility.
Plain pip installations use PyPI, whose Linux Torch build defaults to CUDA 13.0
and requires a compatible 580-series driver unless CUDA 12.6 wheels are installed
first as shown in the README. This replaces the previous CUDA 12.4 contract.

Completed final checks:

- Full release-candidate suite: **74 passed** on macOS/Python 3.13; 10 upstream
  SWIG/SciPy deprecation warnings were observed. The installed inference stack
  matches the root lock. Build-tool installation upgraded auxiliary rich/packaging
  to 15.0.0/26.3; the earlier 70-test suite also passed with the exact locked
  14.2.0/25.0 versions.
- Root and standalone MCP locked dependency exports: **zero known
  vulnerabilities** in refreshed `pip-audit` reports (105 root packages and 34 MCP
  packages, none skipped). These package scans evaluate
  the host-applicable markers; they do not certify Linux CUDA wheels or Debian
  packages.
- Both regenerated lockfiles pass `uv lock --check` after pydicom floors were
  raised to exclude 3.0.0/3.0.1; their resolved pydicom version remains 3.0.2.
- Model tests: **8 passed**, including Apple ZIP metadata compatibility and failed HTTP download,
  checksum validation, and installation preserving a legacy cache. Each of these
  three regressions failed before the cache-preservation fix.
- Independent review approved the final cache replacement and dependency floors.
- Root 1.1.0 wheel/sdist build and pass `twine check`; the installed wheel passes
  console help, public API signature/error/version, nnUNet CPU initialization,
  and synthetic Torch Conv3d smoke checks outside the source tree.
- Standalone MCP source distribution and wheel built during the security audit.
- CUDA 12.6 source/wheel regression verifies the explicit official index and
  Linux x86_64/aarch64 wheel availability in the lock.

## Limits

No GPU inference, Docker build, model download, Windows ACL integration, or
end-to-end image segmentation was performed. GPU numerical behavior and driver
compatibility require deployment validation. Organization tools intentionally
operate with the calling user's filesystem privileges; use private working
folders, because hostile concurrent filesystem changes are outside the tested
plan-validation boundary. Existing URL-only model caches require a verified
replacement before reuse; a failed replacement preserves the old files but
raises rather than executing unverified weights. The telemetry endpoint is
hardcoded HTTPS, so Bandit's scheme warning is not an injectable-URL finding;
server-side telemetry authorization/RLS was not tested.

## Evidence sources

- [Torch memory-corruption advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)
- [Torch 2.13.0 release metadata](https://pypi.org/pypi/torch/2.13.0/json)
- [Torchvision 0.28.0 release metadata](https://pypi.org/pypi/torchvision/0.28.0/json)
- [Pillow 13.0.0 release metadata](https://pypi.org/pypi/pillow/13.0.0/json)
- [Official PyTorch CUDA 12.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v2130)
- [CUDA 12.6 release and driver notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)
- [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- Existing PSMA and FDG release-asset digests were read from the GitHub Releases
  API for `lionz-v.1.0.0` and `lionz-v.1.0.3`, respectively.
