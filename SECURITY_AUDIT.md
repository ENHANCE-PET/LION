# Security audit — 2026-10-31

Scope: LION runtime, release configuration, model downloads, and the standalone
`lion-mcp` package. No clinical documents, model weights, or patient fixtures were
downloaded or added. Changes are local; nothing was pushed or published.

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
- The dormant Windows permission helper no longer grants `Everyone` full control
  over an executable. No active executable download/cache path calls this helper;
  pre-existing filesystem ACLs are not repaired by this change.

## Compatibility and validation

Patched Pillow and Tornado require **Python 3.11+**. The Docker base now uses a
verified, digest-pinned Python 3.12 image. Torch 2.13.0's Linux PyPI metadata
selects CUDA 13.0; runtime checks, release promotion, and the README match this
requirement. Linux GPU deployment requires an NVIDIA 580-series compatible
driver (README minimum 580.65.06). This replaces the previous CUDA 12.4 contract.

Completed checks before the final cache-preservation review:

- Full repository suite: **66 passed** on macOS/Python 3.13 with the locked runtime
  dependencies installed; upstream SWIG deprecation warnings were observed.
- Root and standalone MCP locked dependency exports: **zero known
  vulnerabilities** in separate `pip-audit` reports. These package scans evaluate
  the host-applicable markers; they do not certify Linux CUDA wheels or Debian
  packages.
- Both regenerated lockfiles pass `uv lock --check` after pydicom floors were
  raised to exclude 3.0.0/3.0.1; their resolved pydicom version remains 3.0.2.
- Model tests after the final review: **7 passed**, including failed HTTP download,
  checksum validation, and installation preserving a legacy cache. Each of these
  three regressions failed before the cache-preservation fix.
- Independent review approved the final cache replacement and dependency floors.

Final full-suite execution and refreshed audit reports are being collected by the
coordinating audit process; the counts above identify the checks actually
completed at report-writing time.

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
- [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- Existing PSMA and FDG release-asset digests were read from the GitHub Releases
  API for `lionz-v.1.0.0` and `lionz-v.1.0.3`, respectively.
