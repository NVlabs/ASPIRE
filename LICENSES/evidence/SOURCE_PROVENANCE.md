# Source Provenance

This document records the source-ownership classification used for ASPIRE's
release headers. The machine-readable, per-file inventory is
[`SOURCE_PROVENANCE.tsv`](SOURCE_PROVENANCE.tsv). Reproducible Git blob and
history evidence for every inherited path is recorded separately in
[`SOURCE_MODIFICATION_EVIDENCE.tsv`](SOURCE_MODIFICATION_EVIDENCE.tsv).

The inventory covers every Python and shell source file tracked by the parent
repository after removal of the uncleared AnyGrasp server implementation and
its server-only runtime helper: 401 files (365 Python and 36 shell files).
Git-submodule contents are separate repositories and are governed by their own
licenses.

## Reviewed Git Chain

The evidence fixes every comparison to an immutable commit:

| Role | Repository and commit |
| --- | --- |
| Public CaP-X baseline | `capgym/cap-x@823fcc5dd3e565b45b414f5785668cf32cba13b4` |
| ASPIRE source snapshot | `LRY89757/Holos@1e34c2c53105fb36dd28c49b8f32462cab339330` |
| Byte-identical NVlabs import | `NVlabs/ASPIRE@d27550551354917b807fb794df1c4ce1febf964e` |
| NVlabs pre-header comparison tree | `NVlabs/ASPIRE@a54e872b8d7a2338beb32655bcb3d5f21b54bee7` |
| Legal-header remediation | `NVlabs/ASPIRE@d83065b700a35bfe7c5459eb57c9e4e6ae2adfcf` |
| Reviewed implementation/source base | `NVlabs/ASPIRE@f18052e2aeb24ca3a6cb17c4fc007dbe6276535f` |

All 401 source blobs at the NVlabs import are byte-identical to the same paths
at the ASPIRE source snapshot. Legal headers are not counted as substantive
modifications: inherited paths are compared against the pre-header tree, and
the final reviewed blobs are separately checked to differ from that tree only
by the declared stacked header, except for two files whose required header was
already present.

## Classification Rules

| Classification | Files | Header and terms |
| --- | ---: | --- |
| NVIDIA-authored | 252 | NVIDIA copyright; Apache-2.0 |
| CaP-X with NVIDIA modifications | 91 | Max Fu MIT notice followed by the NVIDIA Apache-2.0 modification block |
| Unchanged CaP-X | 27 | Max Fu MIT notice only |
| PyRoKi through CaP-X | 21 | Chung Min Kim MIT notice and Max Fu MIT notice |
| PyRoKi through CaP-X with NVIDIA modifications | 10 | Chung Min Kim and Max Fu MIT notices followed by the NVIDIA Apache-2.0 modification block |
| **Total** | **401** | |

The exact NVIDIA block is:

```text
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

An NVIDIA block identifies NVIDIA modifications; it does not replace or
relicense upstream material. Complete upstream license texts are retained in
[`LICENSES/`](../), and component-level details are in
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

## CaP-X Boundary and Modification Evidence

The comparison baseline is the public CaP-X release commit
`capgym/cap-x@823fcc5dd3e565b45b414f5785668cf32cba13b4`, authored by Max Fu and
released under MIT. The logical path mapping is:

- baseline `capx/**` to release `aspire/sim/cap/**`;
- baseline `env_configs/**` to release `aspire/sim/env_configs/**`;
- baseline `scripts/**` to release `aspire/sim/scripts/**`; and
- baseline `tests/**` to release `aspire/sim/tests/**`.

Before headers were added, comparison by mapped path and Git blob identified
101 modified inherited files and 48 byte-identical inherited files. The 149
rows consist of 138 mapped simulation files and 11 mapped real-robot PyRoKi
copies. The result exactly matches the declared classifications:

- 91 CaP-X files and 10 PyRoKi-through-CaP-X files are substantively different
  from the exact baseline blob;
- 27 CaP-X files and 21 PyRoKi-through-CaP-X files are byte-identical to the
  exact baseline blob; and
- all 48 byte-identical files carry upstream notices only, while all 101
  modified files carry the upstream notice followed by the NVIDIA
  modification block.

Source-repository history traces the 91 CaP-X differences to
`e2166fc8aeb7e0158dcd8eac3dcf61965ed26da3` and the 10
PyRoKi-through-CaP-X differences to
`8070dda2b8d57e9f70d5604ac5ddffd02d7f2cdb`. Git records Yubo Wu
`<yubowu25@gmail.com>` as author and committer of both source commits. Runyu Lu
`<runyul@nvidia.com>` authored and committed the later NVlabs legal-header
remediation. This is the strongest Git-history evidence; confirmation that the
source contributions are owned or authorized for the NVIDIA modification
notice remains an OSRB/IP-review determination rather than a conclusion
inferred from an email address.

The per-file evidence table records upstream, source, import, pre-header, and
reviewed blob IDs; relevant source and NVlabs commits; author/committer
identities; comparison result; and final-header state. A regeneration must
fail if a mapped path is missing, an expected blob changes, a modified file
becomes identical, an unchanged file differs before headers, or an inventory
classification/count drifts.

Files introduced after that baseline are classified as NVIDIA-authored unless
the repository contains stronger component-specific provenance. In particular,
the visualization helper in
`aspire/sim/cap/integrations/vision/sam3.py` remains CaP-X/Max Fu MIT material;
the surrounding file also carries the NVIDIA modification block because it
changed after the baseline.

## Direct PyRoKi Check

The more specific PyRoKi attribution was independently compared with
`chungmin99/pyroki@95afccc22658c461ab1042a048ae4e9c24bc2a47`.
Twenty-five of the 31 attributed paths map to an exact file at that commit and
were modified while being incorporated into CaP-X. Six paths are
CaP-X-derived extensions for which no direct same-path file exists at the
pinned PyRoKi commit. The evidence table records this distinction rather than
claiming direct byte identity for those six files.

## Reproduction Method

The evidence can be reproduced without trusting filenames or working-tree
timestamps:

1. enumerate inherited rows from `SOURCE_PROVENANCE.tsv`;
2. resolve each recorded upstream, source, import, pre-header, and reviewed
   path with `git rev-parse <commit>:<path>`;
3. compare blob IDs against the values in
   `SOURCE_MODIFICATION_EVIDENCE.tsv`;
4. use `git log --format='%H%x09%an <%ae>%x09%cn <%ce>' -- <path>` in the
   source and NVlabs repositories to verify the recorded history; and
5. strip only the exact declared legal-header prefix from the reviewed blob
   and require the remainder to match the pre-header blob.

The checked repositories were
`https://github.com/capgym/cap-x.git`,
the ASPIRE source-history repository identified in the evidence table,
`https://github.com/chungmin99/pyroki.git`, and
`https://github.com/NVlabs/ASPIRE.git`. A verifier must reject any result other
than 101 modified and 48 byte-identical inherited paths.

The ASPIRE source-history repository may require authorization. The evidence
table retains its exact commit, paths, blobs, commits, and author/committer
identities; OSRB reviewers who need to reproduce the history walk must be
given read access or a separately reviewed Git bundle.

## Real-Robot Boundary

The ASPIRE real-robot implementation is NVIDIA-authored, including
`aspire/real/robot/yam/_base_yam_env.py` and
`aspire/real/robot/yam/yam_sim_env.py`. Earlier comments suggesting an
unidentified “starter code” source were inaccurate and have been removed.

The exceptions are identified third-party material:

- PyRoKi-derived snippets under
  `aspire/real/cap/integrations/motion/pyroki_snippets/` retain the Chung Min
  Kim and CaP-X/Max Fu MIT notices. Modified files add the NVIDIA modification
  block; `_solve_ik_with_manipulability.py` is byte-identical to the CaP-X
  baseline and therefore does not claim an NVIDIA modification.
- Dependency patches under `aspire/real/patches/dependencies/` retain their
  upstream terms and identify Runyu Lu as the author of the NVIDIA ASPIRE
  modifications.
- Robot models, CAD, XML, media, model weights, datasets, and external SDKs are
  not covered by the source-file inventory. Their provenance and open release
  status are tracked in
  [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) and
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Externalized Implementations

- ASPIRE retains its AnyGrasp client protocol and service configuration, but
  does not redistribute the AnyGrasp SDK, model, machine license, copied vendor
  demo server, or server-only runtime helper. An authorized operator supplies
  the service independently.
- ASPIRE retains its Contact-GraspNet service adapter but does not recursively
  distribute the former fork's checkpoints or test data. Source and checkpoint
  paths must be supplied separately under the Contact-GraspNet terms.
- Site-specific calibrated station XML is supplied through
  `YAM_STATION_CALIBRATED_XML`; the recovered calibration patch is not part of
  the public distribution.

## YAM CAD Resolution

The remaining YAM CAD review is resolved:

- `base_visual_gate.stl` was developed internally by NVIDIA. The real-station
  copy and the byte-identical copy in the bundled YAM simulation assets are
  NVIDIA-owned.
- `gripper.stl` was developed internally by NVIDIA.
- `gripper_finger.stl` was exported from the NVIDIA-owned, internally developed
  Fello gripper and is NVIDIA-owned.
- The Stereolabs `zed2i.stl` vendor CAD is no longer redistributed. The active
  ZED station XML and URDF use an NVIDIA-authored axis-aligned box proxy that
  preserves the original CAD bounds, camera body pose, optical frame, and
  calibration. The vendor's detailed model remains available from the
  [official Stereolabs 3D-model page](https://www.stereolabs.com/3dmodels).

The three retained meshes are covered by the project's Apache-2.0 license and
NVIDIA copyright notice. No tracked YAM CAD remains in the unresolved
redistribution category. The adjacent
[`assets/README.md`](../../aspire/real/robot/models/station/assets/README.md) records
the asset-level ownership and ZED reference boundary.
