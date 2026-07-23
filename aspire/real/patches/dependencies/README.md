# Dependency Patch Set

Each subdirectory contains one mail-format patch generated from the final
tracked dependency snapshot at private recovery locator
`c5b71f8a5ded00b33ddce1894ebf1779e763f412`. The NVIDIA ASPIRE modifications
are attributed to `Runyu Lu <runyul@nvidia.com>` and retain the independently
applicable upstream terms. Every patch passed `git apply --check` against the
base tree described below.

| Dependency | Patch base | Verification |
| --- | --- | --- |
| i2rt | `https://github.com/i2rt-robotics/i2rt.git` at `98d177bb511d545c80c0e8ec13ffaf227238a8d6` | The six source preimages match the patch exactly, the public base carries the I2RT Robotics MIT license, and the patch apply-checks cleanly. |
| PyRoki | `https://github.com/chungmin99/pyroki.git` at `95afccc22658c461ab1042a048ae4e9c24bc2a47` | Materialized from the public repository and apply-checked. |
| cuRobo | `https://github.com/NVlabs/curobo.git` at `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (`v0.7.8`) | Materialized from the public repository and apply-checked. NVIDIA license review is still required. |
| RoboCasa | `https://github.com/robocasa/robocasa.git` at `9a3a78680443734786c9784ab661413edb87067b` | Identified by comparing the initial vendor import against public history. It was the nearest exact source state, with only the expected workstation import edits. Apply-checked. |

The reviewed cuRobo patch intentionally omits the workstation snapshot's
absolute `yam_station` symlink because its target contained a private user
home path and could not work in another checkout. The public integration must
resolve station assets through configuration or a repository-relative path.

Apply one patch from the corresponding dependency root with:

```bash
git am /path/to/0001-YAM-workstation-changes-for-DEPENDENCY.patch
```

The patches contain source and repository assets only. Workstation build
outputs and untracked files were not used to generate them.
