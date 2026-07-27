# OSRB Compliance Validation Report

Date: 2026-07-27

Reviewed implementation and source base:
`NVlabs/ASPIRE@f18052e2aeb24ca3a6cb17c4fc007dbe6276535f`

Evidence-only follow-up branch: `compliance/osrb-final-evidence` (use the
revision containing this report for the exact evidence bundle).

## Scope and Policy

This report covers the Apache-2.0 repository remediation, Git-history evidence
for inherited source, recursive-submodule audit, license disclosure, and
parent-source-artifact boundary. It was checked against the OSRB
`osrb-submitter-contribution-repo-compliance-format` skill at
`dea35ad573fcb42247f9dbef7a2374dd655a2617` and its local policy graph at
`d43a1581523fabf0dde1aef71f663f059c6aa6b3`, including the `Apache2 License`
and `Developer Certificate of Origin (DCO)` policy pages.

The project [`LICENSE`](LICENSE) contains the complete Apache License 2.0 text
and the exact 2026 NVIDIA copyright line. [`CONTRIBUTING.md`](CONTRIBUTING.md)
retains the contribution sign-off procedure and full DCO 1.1 text. Nine
retained third-party license files are mapped to their components in the root
[`README.md`](README.md) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Source Header and History Results

The ownership/header inventory is
[`SOURCE_PROVENANCE.tsv`](SOURCE_PROVENANCE.tsv), its rules and comparison
method are in [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md), and per-path Git
blob/history evidence is in
[`SOURCE_MODIFICATION_EVIDENCE.tsv`](SOURCE_MODIFICATION_EVIDENCE.tsv).

| Classification | Files | Result |
| --- | ---: | --- |
| NVIDIA-authored | 252 | Exact NVIDIA Apache-2.0 block present |
| CaP-X with NVIDIA modifications | 91 | Max Fu MIT notice precedes exact NVIDIA block |
| Unchanged CaP-X | 27 | Max Fu MIT notice retained; no NVIDIA claim |
| PyRoKi through CaP-X | 21 | Chung Min Kim and Max Fu MIT notices retained; no NVIDIA claim |
| PyRoKi through CaP-X with NVIDIA modifications | 10 | Both MIT notices precede exact NVIDIA block |
| **Total** | **401** | **All accounted for** |

The immutable comparison chain is:

1. public CaP-X baseline
   `capgym/cap-x@823fcc5dd3e565b45b414f5785668cf32cba13b4`;
2. ASPIRE source
   `LRY89757/Holos@1e34c2c53105fb36dd28c49b8f32462cab339330`;
3. byte-identical NVlabs import
   `NVlabs/ASPIRE@d27550551354917b807fb794df1c4ce1febf964e`;
4. pre-header comparison tree
   `NVlabs/ASPIRE@a54e872b8d7a2338beb32655bcb3d5f21b54bee7`;
5. legal-header commit
   `NVlabs/ASPIRE@d83065b700a35bfe7c5459eb57c9e4e6ae2adfcf`; and
6. reviewed implementation/source base
   `NVlabs/ASPIRE@f18052e2aeb24ca3a6cb17c4fc007dbe6276535f`.

All 401 source blobs at the import match the ASPIRE source snapshot. Comparing
the 149 inherited paths before legal headers independently proves 101
substantive blob differences and 48 byte-identical paths, exactly matching the
declared stacked/unchanged classifications. Regenerating the evidence table
from Git produced the same file byte-for-byte after LF normalization; its
SHA-256 is
`db01b56526310aae25fc5cfea84878590f8098f5a7d649294b845823e2a0dfd3`.

Source history traces the 91 CaP-X changes to
`e2166fc8aeb7e0158dcd8eac3dcf61965ed26da3` and the 10
PyRoKi-through-CaP-X changes to
`8070dda2b8d57e9f70d5604ac5ddffd02d7f2cdb`. Git records Yubo Wu
`<yubowu25@gmail.com>` as author and committer of both source commits. Runyu Lu
`<runyul@nvidia.com>` authored and committed the NVlabs legal-header
remediation. Corporate ownership/authorization of those source contributions
remains an OSRB/IP-review determination; this report does not infer it from an
email domain.

The direct PyRoKi comparison uses
`chungmin99/pyroki@95afccc22658c461ab1042a048ae4e9c24bc2a47`: 25 attributed
paths map to a direct file and were modified in CaP-X, while six are
CaP-X-derived extensions with no direct same-path file at that pin.

## Third-Party Remediation Results

- The copied/demo-derived AnyGrasp server and server-only helper are removed.
  ASPIRE retains its client protocol and an external-service launcher only.
- The Contact-GraspNet recursive submodule is removed. Source and checkpoints
  must be supplied separately through the documented environment variables.
- The recovered calibration patch is removed; calibrated station XML remains
  operator-local through `YAM_STATION_CALIBRATED_XML`.
- All four dependency patches identify `Runyu Lu <runyul@nvidia.com>`, an exact
  reachable upstream base, and the upstream license.
- The i2rt patch base is the public MIT-licensed
  `i2rt-robotics/i2rt@98d177bb511d545c80c0e8ec13ffaf227238a8d6`.
- The malformed `model2__1.stl` payload is absent.
- YAM CAD review is resolved as described below.

## Populated Submodule Results

All six parent gitlinks were recursively initialized, resolved to the exact
pin, and had clean working trees. The full inventory and immutable license-file
hashes are in [`SUBMODULE_AUDIT.md`](SUBMODULE_AUDIT.md) and
[`SUBMODULE_AUDIT.tsv`](SUBMODULE_AUDIT.tsv).

| Submodule | Exact pin | Key result |
| --- | --- | --- |
| LIBERO-PRO | `47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a` | Code MIT; datasets described as CC BY 4.0; object-asset coverage requires OSRB review |
| Robosuite (YAM fork) | `97292732ed909ac3ae116579fb768607034a4dbd` | MIT/MuJoCo attribution; extensive YAM and robot assets require provenance review |
| Robosuite (LIBERO dependency) | `a498b087d4bc5a3981e3d27030d09bc537a537f3` | MIT/MuJoCo attribution; bundled robot/arena assets require provenance review |
| SAM 3 | `6fe87d64a5beb9084923d7a9e002741178635b09` | Custom SAM License; no tracked model weights at pin |
| cuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | Custom NVIDIA terms plus `LICENSE_ASSETS`; no current LFS pointers |
| `b1k` | `272ec5ca9936453c4a8fd335c4dfba61245e33ca` | Mixed terms; Pixar HumanFemale assets prohibit redistribution without written authorization |

The parent and current submodule trees contain zero current Git LFS entries.
cuRobo has historical LFS records but no pointer in its pinned current tree.
`b1k/OmniGibson/.gitmodules` declares five URLs but has zero corresponding
nested gitlinks, so no nested repository was fetched.

The parent ASPIRE source artifact stores gitlinks only and does not include
populated submodule contents. Recursive checkout fetches those repositories
separately under their own terms. This boundary prevents the parent archive
from containing the restricted submodule payloads, but it does not itself
approve custom-license dependencies or recursive redistribution.

## Parent Source Artifact

The release artifact is defined as a Git source archive of the parent
repository only:

- all parent-tree blobs, `.gitmodules`, licenses, notices, provenance tables,
  and audit reports are included;
- the candidate Git tree retains all six gitlinks, while the tar format cannot
  encode mode-160000 entries; the companion manifest therefore records their
  exact paths and object IDs;
- populated submodule content and `.git` history are excluded;
- no current Git LFS object or pointer is required by the parent tree; and
- the artifact manifest records the exact tree/revision, complete path and
  size inventory, and SHA-256.

The final artifact filename, SHA-256, and scan result are recorded in
`ASPIRE_OSRB_RELEASE_ARTIFACT_MANIFEST.txt` delivered alongside this report.

## Validation Results

| Check | Result |
| --- | --- |
| Python parsing | 365 files parsed with `ast.parse`; zero errors |
| Shell parsing | 36 files passed `bash -n` |
| Header manifest | 401 files; exact classification counts; zero mismatches |
| Import identity | 401/401 source blobs match the ASPIRE source snapshot |
| Inherited baseline comparison | 149 paths: 101 modified, 48 byte-identical; zero mismatches |
| Evidence regeneration | Byte-identical after LF normalization; SHA-256 recorded above |
| Dependency patches | i2rt, PyRoKi, RoboCasa, and cuRobo apply-check cleanly at the recorded exact bases |
| Populated submodules | Six exact clean pins; all licenses, assets, nested metadata, and LFS state inventoried |
| Local licenses and notices | Root Apache-2.0, NOTICE, DCO, and all nine retained license files present |
| Parent artifact restricted-content scan | No populated SAM 3, cuRobo, `b1k`, Pixar, Robosuite, or LIBERO-PRO payload |
| Secret/private-key/internal-URL scan | Zero confirmed credentials or private keys in the parent artifact |
| Parent artifact asset inventory | 21 YAM-station STL, 5 station USD, 8 Franka STL, 59 Franka OBJ, and 1 project GIF; all mapped to existing provenance/notice records |
| Markdown local links | 337 targets checked; zero missing |
| External documentation links | 38 current targets reached successfully, including immutable license/source links |
| Whitespace | `git diff --check` passes |

No GPU, simulator, hardware, station service, robot RPC, or physical-robot test
was run. The follow-up changes documentation and evidence only.

## YAM CAD Resolution

- `base_visual_gate.stl` and `gripper.stl` were developed internally by NVIDIA.
- `gripper_finger.stl` was exported from the NVIDIA-owned, internally developed
  Fello gripper.
- The Stereolabs `zed2i.stl` file was removed. Both ZED station XML variants and
  the ZED station URDF use an NVIDIA-authored box proxy matching the removed
  mesh's axis-aligned bounds. Camera pose, optical frame, and calibration are
  unchanged. The detailed vendor CAD is referenced only through the
  [official Stereolabs download page](https://www.stereolabs.com/3dmodels).

No parent-tree YAM CAD remains in the unresolved redistribution category.

## External Release Gates

Repository formatting and the parent-artifact boundary do not constitute final
release authorization. Before release, OSRB/Legal must decide:

1. whether the Git-history/contribution record supports the 101 NVIDIA
   modification notices, and whether reviewers need authorized source-history
   access or a separately reviewed Git bundle;
2. whether gitlink-only references to SAM 3, cuRobo, and `b1k` are acceptable;
3. whether release instructions may recommend recursive checkout of `b1k`
   while its pin contains the no-redistribution Pixar asset;
4. whether LIBERO-PRO and both Robosuite forks adequately establish terms for
   their bundled datasets, robot descriptions, CAD, and object assets; and
5. whether the intended use is approved under the SAM 3 and cuRobo custom
   licenses.

Closest-VP approval, ongoing IP-review-process compliance, and final release
authorization remain separate external gates. If OSRB/Legal rejects a
gitlink-only reference, the documented safe fallback is to remove that
gitlink, retain the exact tested revision in setup documentation, and require
users to obtain the dependency separately.
