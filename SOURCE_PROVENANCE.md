# Source Provenance

This document records the source-ownership classification used for ASPIRE's
release headers. The machine-readable, per-file inventory is
[`SOURCE_PROVENANCE.tsv`](SOURCE_PROVENANCE.tsv).

The inventory covers every Python and shell source file tracked by the parent
repository after removal of the uncleared AnyGrasp server implementation and
its server-only runtime helper: 401 files (365 Python and 36 shell files).
Git-submodule contents are separate repositories and are governed by their own
licenses.

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
[`LICENSES/`](LICENSES/), and component-level details are in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## CaP-X Boundary

The comparison baseline is the public CaP-X release commit
`LRY89757/Holos@823fcc5dd3e565b45b414f5785668cf32cba13b4`, authored by Max Fu and
released under MIT. The logical path mapping is:

- baseline `capx/**` to release `aspire/sim/cap/**`;
- baseline `env_configs/**` to release `aspire/sim/env_configs/**`;
- baseline `scripts/**` to release `aspire/sim/scripts/**`; and
- baseline `tests/**` to release `aspire/sim/tests/**`.

Before headers were added, comparison by mapped path and Git blob identified
47 byte-identical CaP-X files and 91 modified CaP-X files. Twenty of the 47
byte-identical files contain PyRoKi-derived snippets and therefore use the more
specific PyRoKi-plus-CaP-X attribution in the table above. The per-file TSV
preserves the original mapped baseline path for every inherited file.

Files introduced after that baseline are classified as NVIDIA-authored unless
the repository contains stronger component-specific provenance. In particular,
the visualization helper in
`aspire/sim/cap/integrations/vision/sam3.py` remains CaP-X/Max Fu MIT material;
the surrounding file also carries the NVIDIA modification block because it
changed after the baseline.

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
  status are tracked in `THIRD_PARTY_LICENSES.md` and
  `THIRD_PARTY_NOTICES.md`.

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

## Remaining Release Gate

The source-file classification does not clear the provenance of these YAM CAD
meshes:

- `base_visual_gate.stl`
- `gripper.stl`
- `gripper_finger.stl`
- `zed2i.stl`

They must receive source/owner and redistribution confirmation, or be removed,
externalized, or replaced before the release is declared cleared.
