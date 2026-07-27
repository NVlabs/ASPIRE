# Git Submodule Audit

This audit covers every Git submodule declared by the parent ASPIRE repository
at `f18052e2aeb24ca3a6cb17c4fc007dbe6276535f`. The machine-readable inventory is
[`SUBMODULE_AUDIT.tsv`](SUBMODULE_AUDIT.tsv).

The parent ASPIRE source artifact stores `.gitmodules` and six gitlinks only.
It does not contain populated submodule files. Running
`git submodule update --init --recursive` fetches the repositories separately
from their configured remotes, and their own licenses govern the fetched
source, models, datasets, robot descriptions, CAD, media, and other assets.

## Exact Parent Gitlinks

| Submodule | Configured source | Exact gitlink SHA | Principal terms | Recursive-checkout finding |
| --- | --- | --- | --- | --- |
| LIBERO-PRO | [`uynitsuj/LIBERO-PRO`](https://github.com/uynitsuj/LIBERO-PRO) | `47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a` | Code: MIT; README describes datasets as CC BY 4.0 | Contains LIBERO datasets and object assets, including paths named `turbosquid_objects`; confirm that the upstream dataset statement covers every redistributed asset. |
| Robosuite (YAM fork) | [`uynitsuj/robosuite`](https://github.com/uynitsuj/robosuite) | `97292732ed909ac3ae116579fb768607034a4dbd` | MIT, with retained DeepMind MuJoCo Apache-2.0 attribution | Contains extensive YAM and other robot meshes/descriptions. Confirm asset provenance and redistribution terms independently of the code-level MIT license. |
| Robosuite (LIBERO dependency) | [`Max-Fu/robosuite`](https://github.com/Max-Fu/robosuite) | `a498b087d4bc5a3981e3d27030d09bc537a537f3` | MIT, with retained DeepMind MuJoCo Apache-2.0 attribution | Contains robot and arena assets. Confirm that asset terms are covered by the upstream distribution. |
| SAM 3 | [`Max-Fu/sam3`](https://github.com/Max-Fu/sam3) | `6fe87d64a5beb9084923d7a9e002741178635b09` | Custom SAM License | The pin contains source, evaluation material, and sample media but no tracked model weights. Redistribution, acknowledgment, trade-control, and prohibited-use terms remain applicable. |
| cuRobo | [`NVlabs/curobo`](https://github.com/NVlabs/curobo) | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (`v0.7.8`) | Custom NVIDIA license; public use limited to noncommercial research or evaluation, plus `LICENSE_ASSETS` | The pin contains robot and scene assets governed by component-specific terms. The current tree contains no Git LFS pointers; historical LFS use does not add payloads to the parent artifact. |
| BEHAVIOR-1K bundle (`b1k`) | [`qingh097/b1k`](https://github.com/qingh097/b1k) | `272ec5ca9936453c4a8fd335c4dfba61245e33ca` | Mixed MIT, Apache-2.0, and asset-specific terms | Contains Pixar HumanFemale USD assets whose bundled license prohibits redistribution without Pixar's written authorization. Treat recursive distribution as blocked pending OSRB/Legal confirmation. |

All six populated working trees resolved to the exact parent gitlink SHA and
were clean at the time of review.

## License and Asset Findings

### LIBERO-PRO

- Root [`LICENSE`](https://github.com/uynitsuj/LIBERO-PRO/blob/47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a/LICENSE)
  is MIT.
- The upstream README describes code as MIT and datasets as CC BY 4.0.
- The pin contains 2,608 tracked files and approximately 680 MB of Git blob
  content, including dataset/task material and object assets under
  `stable_scanned_objects`, `stable_hope_objects`, and `turbosquid_objects`.
- No per-asset license was found for the `turbosquid_objects` paths. The parent
  artifact does not include these files, but OSRB/Legal should confirm whether
  the upstream dataset statement is sufficient for a recursive distribution.

### Robosuite forks

- Both pins retain an MIT license plus the DeepMind MuJoCo Apache-2.0
  attribution.
- The `uynitsuj` fork contains 1,528 tracked files and approximately 876 MB of
  Git blob content, including YAM station XML, URDF, camera, and mesh assets.
- The `Max-Fu` fork contains 1,104 tracked files and approximately 639 MB of
  Git blob content, including robot and arena assets.
- A repository-level software license does not by itself prove the provenance
  of every CAD or robot asset. Their recursive distribution remains subject to
  OSRB review, especially the YAM asset collection in the `uynitsuj` fork.

### SAM 3

- The exact pin retains the custom [SAM License](https://github.com/Max-Fu/sam3/blob/6fe87d64a5beb9084923d7a9e002741178635b09/LICENSE).
- It contains 504 tracked files and approximately 72 MB of Git blob content.
- No model checkpoint or weight file and no Git LFS pointer is tracked at the
  pin. Gated model assets are obtained separately.
- The license's redistribution, acknowledgment, trade-control, and
  prohibited-use conditions still govern the separately fetched source and
  included media/evaluation material.

### cuRobo

- The exact pin retains the custom [cuRobo license](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE)
  and an [asset-license inventory](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE_ASSETS).
- It contains 568 tracked files and approximately 181 MB of Git blob content,
  including robot and scene assets.
- The current pin has no Git LFS pointers. Historical commits contain LFS
  records, but those historical objects are not present in the parent release
  artifact.
- Public-recipient use is limited to noncommercial research or evaluation;
  approval of this custom-license dependency remains separate from ASPIRE's
  Apache-2.0 repository compliance.

### BEHAVIOR-1K bundle (`b1k`)

- The pin contains 19,507 tracked files and approximately 940 MB of Git blob
  content.
- OmniGibson, BDDL, and JoyLo retain MIT license files; Panda assets retain
  Apache-2.0 license files.
- `asset_pipeline/b1k_pipeline/tools/HumanFemale/` contains Pixar
  HumanFemale USD assets and a license permitting personal noncommercial USD
  testing but prohibiting distribution of the asset or derivatives without
  written authorization.
- `OmniGibson/.gitmodules` declares five repositories, but the pinned tree
  contains no corresponding nested gitlinks. They were not fetched by the
  recursive update and are not part of the parent artifact.
- The parent ASPIRE artifact includes only the `b1k` gitlink. A release
  procedure that vendors, mirrors, packages, or otherwise redistributes the
  populated `b1k` checkout must not proceed without OSRB/Legal resolution of
  the Pixar restriction.

## Git LFS and Nested-Repository Result

- The parent repository and every current submodule tree contain zero current
  Git LFS entries.
- cuRobo has historical LFS records, but no pointer is tracked at the pinned
  current tree.
- The six parent gitlinks are the complete recursive gitlink inventory.
- The five URLs declared by `b1k/OmniGibson/.gitmodules` have no gitlinks at
  the pinned commit and therefore are metadata only, not fetched dependencies.

## Release Boundary and Open Gates

The parent source archive may be released without submodule contents only if
its manifest makes the gitlink-only boundary explicit. The following remain
external approval questions:

1. whether gitlink-only references to SAM 3, cuRobo, and `b1k` are acceptable;
2. whether any release instruction may recommend recursive checkout of `b1k`
   while the pinned tree contains the no-redistribution Pixar asset;
3. whether the LIBERO-PRO dataset statement covers all included object assets;
4. whether both Robosuite forks adequately establish terms for their bundled
   CAD and robot-description assets; and
5. whether custom-license SAM 3 and cuRobo dependencies are approved for the
   intended release.

If OSRB/Legal rejects a gitlink-only reference, the safe fallback is to remove
that gitlink, document the tested upstream revision, and require users to
obtain the dependency separately. That structural change is intentionally not
made in this evidence-only follow-up.
