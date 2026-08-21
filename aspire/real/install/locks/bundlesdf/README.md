# BundleSDF Recovery Material

## Source snapshot

The recovery bundle included a deterministic source-only archive from private
snapshot `c5b71f8a5ded00b33ddce1894ebf1779e763f412`. That archive is intentionally
not included in this repository because its redistribution rights are
unresolved.

SHA-256:

```text
3b44e53a774ce5761b24f19c999a0c7ba1d1d0d376d8b4ffbf8d323f568eac24
```

The archive excludes `libs/`, build directories, `__pycache__`, Python byte
code, shared objects, and LoFTR checkpoint files. It is a recovery artifact
for source and legal review, not an approval to redistribute.

## Provenance and license findings

- The high-level tracker states that it was ported from NVIDIA BundleSDF:
  `https://github.com/NVlabs/BundleSDF`.
- `bundlesdf.py`, `Utils.py`, `tool.py`, `offscreen_renderer.py`,
  `run_live_bundlesdf.py`, and the copied BundleTrack C++/CUDA tree carry
  NVIDIA copyright headers saying use or distribution without an express
  NVIDIA license agreement is strictly prohibited.
- The BundleTrack implementation is copied into `BundleTrack/src/`, but the
  package root has no LICENSE or source commit identifying the exact NVIDIA
  BundleSDF/BundleTrack revision.
- `BundleTrack/LoFTR` comes from `https://github.com/zju3dv/LoFTR` and includes
  an Apache-2.0 LICENSE. The exact upstream commit was not retained.
- The LoFTR README also references Magic Leap's
  `SuperGluePretrainedNetwork` and warns that its stricter license prevents
  direct redistribution. Review `BundleTrack/LoFTR/demo/utils.py` and related
  utility files before publication.
- SAM2 and SAM3 model implementations are not copied into this archive. The
  integration calls Hugging Face Transformers and downloads
  `facebook/sam2.1-hiera-large` and gated `facebook/sam3` model assets. The
  station cache used revisions `665f8e2ad61cf5f53d65644ff27c8ee525124610`
  and `3c879f39826c281e95690f02c7821c4de09afae7`, respectively.

Public redistribution is blocked until the BundleSDF/BundleTrack source base
and license grant are established and all copied utility code is audited.

## Locked build definition

`build-lock.env` pins Miniforge and both OpenCV source archives by immutable
version/commit and SHA-256. `conda-linux-64.explicit.txt` records all 300
recovered conda-forge packages with package MD5s. `conda-solve.json` preserves
the full 2026-07-02 solver result, including package SHA-256 values and license
metadata.

The Miniforge installer and both OpenCV archives were downloaded from their
recorded URLs during recovery and independently matched the SHA-256 values in
`build-lock.env`.

`conda-solve.json` has one sanitation-only change: the dry-run prefix was
rewritten from the workstation home directory to
`/opt/conda/envs/bundlesdf-build`. Package resolution data is unchanged. Its
SHA-256 is
`9589d384879c77af07299d81d4c8a2d6144e407c484bd0c8f772db10990e055c`.

The recovered metadata records these intended build inputs:

```text
GCC/G++ 11.4.0
sysroot_linux-64 2.17
CMake 4.3.4
Ninja 1.13.2
Python 3.11.15
PCL 1.15.1
OpenCV/opencv_contrib 4.11.0 built from the pinned commits
Host CUDA toolkit 12.8.93
CUDA architectures 80, 86, 89, 90, 120
```

This list is not a claim that the explicit environment is internally
consistent. Do not recreate the environment directly from the recovered
explicit file: it contains `cuda-version=13.3` and a CUDA 13 `viskores` build,
contradicting the CUDA 12.8 host/toolchain used for BundleTrack. It is retained
as evidence of the recovered solver state, not as an executable release lock.

The integrated `compile_bundlesdf.sh` sources `build-lock.env`, verifies
downloads, and extracts pinned OpenCV archives. It requires a separate reviewed
CUDA 12.8-compatible conda explicit spec and checksum before creating a build
environment.
