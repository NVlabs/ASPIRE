# Third-Party Notices

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The Apache License 2.0 in [`LICENSE`](../LICENSE) applies to ASPIRE material whose
copyright owners have authorized that license. It does not relicense inherited
code, modified third-party code, dependency patches, Git submodules, models,
datasets, robot descriptions, CAD, media, SDKs, or other assets. Those
materials retain their original terms.

This repository contains modifications of or dependencies on:

- CaP-X-derived code (MIT)
- PyRoKi snippets, assets, and patches (MIT)
- Hydra utility code (MIT)
- RoboCasa patches (MIT, with DeepMind MuJoCo Apache-2.0 attribution)
- i2rt YAM robot-model assets (MIT)
- Trossen ALOHA camera-model assets (BSD-3-Clause)
- Franka Emika Panda robot-model assets (Apache-2.0)
- LIBERO-PRO (MIT)
- Robosuite forks (MIT, with DeepMind MuJoCo Apache-2.0 attribution)
- Contact-GraspNet (custom NVIDIA license; noncommercial public-use limit)
- SAM 3 (custom Meta SAM License)
- cuRobo v0.7.8 (custom NVIDIA license; noncommercial public-use limit)
- BEHAVIOR-1K components and assets (mixed terms)

## Incorporated and modified material

- Substantial portions of the repository descend from the CaP-X import and
  retain the MIT notice of Max Fu in
  [`MIT-CaP-X.txt`](MIT-CaP-X.txt).
- PyRoKi-derived snippets and assets are present under
  `aspire/sim/cap/third_party/pyroki_snippets/`,
  `aspire/sim/cap/integrations/motion/pyroki_snippets/`,
  `aspire/real/cap/integrations/motion/pyroki_snippets/`, and
  `aspire/sim/cap/serving/assets/panda_spheres.json`. They retain the MIT
  notice of Chung Min Kim.
- `aspire/sim/cap/envs/configs/instantiate.py` contains an adapted Hydra
  implementation and retains the MIT notice of Facebook, Inc. and its
  affiliates.
- Dependency patches under `aspire/real/patches/dependencies/` contain upstream
  context and modified material from PyRoKi, RoboCasa, cuRobo v0.7.8, and a
  public MIT-licensed i2rt tree. Their upstream terms apply independently;
  the patch metadata identifies Runyu Lu as author of the NVIDIA ASPIRE
  modifications and records an exact upstream base.
- The Franka meshes and modified MJCF under
  `aspire/sim/cap/envs/assets/franka_pick_place/` derive from the Apache-2.0
  Franka model in the pinned MuJoCo Menagerie tree. The local README and XML
  identify the source and ASPIRE modifications.
- Seventeen `model2*` station meshes are from the MIT-licensed i2rt YAM model,
  and `d405.stl` is from the BSD-3-Clause Trossen ALOHA model. Their complete
  notices are retained in this directory.
- The local YAM station XML files are modified from the pinned MIT-licensed
  Robosuite fork. The retained `base_visual_gate.stl`, `gripper.stl`, and
  `gripper_finger.stl` assets are NVIDIA-owned and internally developed. The
  former Stereolabs ZED 2i vendor mesh is not redistributed; the station models
  use an NVIDIA-authored box proxy, and the detailed vendor CAD is available
  from the [official Stereolabs 3D-model page](https://www.stereolabs.com/3dmodels).
  The fork's complete MIT and MuJoCo attribution notice is retained in
  [`MIT-Robosuite.txt`](MIT-Robosuite.txt).

## Git submodules and external components

The parent ASPIRE source artifact stores seven gitlinks and `.gitmodules`
metadata only; it does not include populated submodule contents. Initializing
the gitlinks fetches source, models, test data, robot descriptions, CAD, media,
and assets separately from their configured repositories.

The populated-tree audit found the following release-relevant conditions:

- the pinned cuRobo version is limited to noncommercial research or evaluation
  for public recipients and contains assets listed separately in
  `LICENSE_ASSETS`;
- SAM 3 uses the custom Meta SAM License; the exact pin contains source and
  sample/evaluation media but no tracked model weights;
- Contact-GraspNet uses the custom NVIDIA source license; its pinned PyTorch
  upstream includes a checkpoint, test data, and media, and ASPIRE applies the
  documented three-hunk compatibility patch;
- the pinned `b1k` tree contains mixed terms and Pixar HumanFemale USD assets
  whose license prohibits redistribution without written authorization;
- LIBERO-PRO describes its code as MIT and datasets as CC BY 4.0, but contains
  object-asset collections, including `turbosquid_objects`, for which no
  separate per-asset license was found; and
- both Robosuite forks contain substantial robot-description and CAD/mesh
  assets whose provenance must be reviewed independently of their
  repository-level MIT license.

These findings do not place the restricted files in the parent source
artifact. They do mean that custom-license approval and any procedure that
recommends or redistributes a populated recursive checkout remain separate
OSRB/Legal release gates.

BundleSDF source and checkpoints, AnyGrasp vendor source/server/model/license
material, SAM model weights,
Stereolabs ZED SDK material, and the detailed Stereolabs ZED 2i reference CAD
are intentionally external and retain their vendor or upstream terms. SAM 2.1
is Apache-2.0; SAM 3 uses the custom SAM License and gated-access conditions.

For exact paths, revisions, immutable license links, component distribution
boundaries, license hashes, Git LFS results, and nested-repository findings,
see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) and
[`SUBMODULE_AUDIT.md`](evidence/SUBMODULE_AUDIT.md).

The exact source-file ownership boundary, including unchanged CaP-X source,
CaP-X source modified by NVIDIA, and NVIDIA-authored source, is recorded in
[`SOURCE_PROVENANCE.md`](evidence/SOURCE_PROVENANCE.md).
