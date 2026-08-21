# Installer Set Review

Source snapshot locator:
`c5b71f8a5ded00b33ddce1894ebf1779e763f412`.

The recovery bundle's four shell files passed `bash -n`. Three are integrated
under `install/`. The recovered `tools/runtime_env.sh` was not imported because
the repository already has a sanitized runtime helper and the recovered copy
contained two local defaults:

- the operator-specific AnyGrasp license ZIP filename;
- the private cuRobo SSH host alias.

The following release work remains:

- `install_cap.sh` now requires reviewed local dependency sources and does not
  fetch private Git LFS payloads.
- AnyGrasp SDK files, checkpoints, native extensions, and authorization files
  must be obtained from the vendor and must not be placed in a public archive.
- `compile_bundlesdf.sh` now verifies pinned Miniforge and OpenCV downloads. It
  refuses to create the build environment until a separately reviewed,
  checksummed CUDA 12.8 explicit conda spec is supplied; the recovered solve
  contains conflicting CUDA 13 packages and is retained only for audit.
- `build_zed_sdk.sh` downloads and runs Stereolabs' official installer, which
  has its own terms and writes to `/usr/local/zed`.
- The installer interactively stores credentials in `.forge_env.secrets`.
  No secret values are present in this archive.
- The scripts assume Ubuntu 22.04, Python 3.11, CUDA 12.8, and a compatible
  NVIDIA driver.

Review the remaining limitations in `README.md` before running the installer.
In particular, no approved BundleSDF source, AnyGrasp payload, calibration
model, or vendor authorization material is included here.
