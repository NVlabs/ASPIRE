# OSRB Compliance Validation Report

Date: 2026-07-22

Reviewed base: `NVlabs/ASPIRE@0defc580ef18cf9e6c9d1fd2e0c5e531788efcca`

## Scope and Policy

This report covers the settled Apache-2.0 repository remediation. It was
checked against the OSRB
`osrb-submitter-contribution-repo-compliance-format` skill at
`dea35ad573fcb42247f9dbef7a2374dd655a2617` and its local policy graph at
`d43a1581523fabf0dde1aef71f663f059c6aa6b3`, including the `Apache2 License`
and `Developer Certificate of Origin (DCO)` policy pages.

The project [`LICENSE`](LICENSE) contains the complete Apache License 2.0 text
and the exact 2026 NVIDIA copyright line. [`CONTRIBUTING.md`](CONTRIBUTING.md)
retains the contribution sign-off procedure and full DCO 1.1 text.

## Source Header Results

The per-file source inventory is [`SOURCE_PROVENANCE.tsv`](SOURCE_PROVENANCE.tsv),
with classification rules in [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md).
It accounts for all 401 Python and shell files remaining after removal of two
uncleared AnyGrasp implementation files.

| Classification | Files | Result |
| --- | ---: | --- |
| NVIDIA-authored | 252 | Exact NVIDIA Apache-2.0 block present |
| CaP-X with NVIDIA modifications | 91 | Max Fu MIT notice precedes exact NVIDIA block |
| Unchanged CaP-X | 27 | Max Fu MIT notice retained; no NVIDIA claim |
| PyRoKi through CaP-X | 21 | Chung Min Kim and Max Fu MIT notices retained; no NVIDIA claim |
| PyRoKi through CaP-X with NVIDIA modifications | 10 | Both MIT notices precede exact NVIDIA block |
| **Total** | **401** | **All accounted for** |

Automated header validation found 353 NVIDIA blocks, 149 Max Fu notices, and
31 Chung Min Kim notices, with zero classification/header mismatches. A direct
comparison against `LRY89757/Holos@823fcc5dd3e565b45b414f5785668cf32cba13b4`
validated all 138 CaP-X-inherited files. The real-tree
`_solve_ik_with_manipulability.py` copy was also verified byte-identical to that
baseline and therefore carries only its two upstream MIT notices.

## Third-Party Remediation Results

- The copied/demo-derived AnyGrasp server and server-only helper are removed.
  ASPIRE retains its client protocol and an external-service launcher only.
- The Contact-GraspNet recursive submodule is removed. Source and checkpoints
  must be supplied separately through the documented environment variables.
- The recovered calibration patch is removed; calibrated station XML remains
  operator-local through `YAM_STATION_CALIBRATED_XML`.
- All four dependency patches identify `Runyu Lu <runyul@nvidia.com>`, an exact
  reachable upstream base, and the upstream license. Each patch passed
  `git apply --check` against that base.
- The i2rt patch base is the public MIT-licensed
  `i2rt-robotics/i2rt@98d177bb511d545c80c0e8ec13ffaf227238a8d6`;
  [`LICENSES/MIT-i2rt.txt`](LICENSES/MIT-i2rt.txt) is byte-identical to its
  upstream license.
- The malformed `model2__1.stl` payload is absent.
- The resulting parent repository has six intentional gitlinks. Every remote
  and pinned revision was checked as reachable.

The resulting notices and distribution boundaries are recorded in
[`NOTICE`](NOTICE), [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Validation Results

| Check | Result |
| --- | --- |
| Python parsing | 365 files parsed with `ast.parse`; zero errors |
| Shell parsing | 36 files passed `bash -n` |
| Header manifest | 401 files; zero mismatches |
| CaP-X baseline comparison | 138 inherited paths; zero classification mismatches |
| Dependency patches | i2rt, PyRoKi, RoboCasa, and cuRobo apply-check cleanly |
| AnyGrasp external-service gate | Missing external service stops with an actionable error |
| TOML and lock consistency | `pyproject.toml` and `uv.lock` parse; no bundled Contact-GraspNet package/path remains |
| Markdown local links | 61 checked targets across changed documentation; zero missing |
| Changed external license links | Contact-GraspNet, i2rt, PyRoKi, RoboCasa, and cuRobo returned HTTP 200 |
| Secret-prefix/private-key scan | Zero findings |
| Whitespace | `git diff --check` passes |

`uv lock --check` reaches cuRobo metadata resolution but cannot complete on the
macOS ARM validation host because the project intentionally supports Linux
x86-64 CUDA wheels. The Contact-GraspNet lock delta was therefore validated by
TOML parsing and direct consistency checks: the removed editable package,
optional dependency, source path, and extra build dependency are absent from
both project and lock data. No GPU, simulator, hardware, or physical-robot test
was run: the changed integration behavior is limited to refusing bundled
payloads and requiring explicitly supplied external services and paths.

## Open Release Gates

This report does **not** declare the public release cleared. Written source,
owner, and redistribution confirmation is still required for:

- `aspire/real/robot/models/station/assets/base_visual_gate.stl`
- `aspire/real/robot/models/station/assets/gripper.stl`
- `aspire/real/robot/models/station/assets/gripper_finger.stl`
- `aspire/real/robot/models/station/assets/zed2i.stl`

If confirmation is unavailable, those meshes must be removed, externalized,
or replaced before release. The closest-VP approval in the OSRB bug is a
separate external gate.
