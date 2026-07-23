# Third-Party Licenses

ASPIRE contains, modifies, or integrates the third-party material listed below.
The project-level [Apache License 2.0](LICENSE) does not replace any
component-specific terms. Exact revisions are recorded whenever this repository
pins or modifies a particular upstream version.

This catalog focuses on material present in the parent source tree, dependency
patches, pinned Git submodules, and external components with material license
constraints. Python packages installed separately are declared in
`aspire/sim/pyproject.toml` and `aspire/real/pyproject.toml`; their own license
files and package metadata apply. This is not a transitive software bill of
materials.

## Incorporated or modified material

| No. | Component | Form and reviewed revision | License | License link |
| ---: | --- | --- | --- | --- |
| 1 | CaP-X-derived code | Inherited parent-tree source; original ASPIRE repository baseline `823fcc5dd3e565b45b414f5785668cf32cba13b4` | MIT | [Local license](LICENSES/MIT-CaP-X.txt) |
| 2 | PyRoKi | Embedded snippets and assets, plus a dependency patch based on `chungmin99/pyroki@95afccc22658c461ab1042a048ae4e9c24bc2a47` | MIT | [Local license](LICENSES/MIT-PyRoKi.txt), [upstream license](https://github.com/chungmin99/pyroki/blob/95afccc22658c461ab1042a048ae4e9c24bc2a47/LICENSE) |
| 3 | Hydra | Adapted `_locate` implementation from `facebookresearch/hydra@57690d7c4e8b5e88dad07d67278f613a739e6d13` | MIT | [Local license](LICENSES/MIT-Hydra.txt), [upstream license](https://github.com/facebookresearch/hydra/blob/57690d7c4e8b5e88dad07d67278f613a739e6d13/LICENSE) |
| 4 | RoboCasa | Dependency patch based on `robocasa/robocasa@9a3a78680443734786c9784ab661413edb87067b` | MIT, with retained DeepMind MuJoCo Apache-2.0 attribution | [Local license](LICENSES/MIT-RoboCasa.txt), [upstream license](https://github.com/robocasa/robocasa/blob/9a3a78680443734786c9784ab661413edb87067b/LICENSE) |
| 5 | i2rt source patch | NVIDIA ASPIRE modifications based on `i2rt-robotics/i2rt@98d177bb511d545c80c0e8ec13ffaf227238a8d6` | MIT upstream; NVIDIA modifications identified in the patch metadata | [Local license](LICENSES/MIT-i2rt.txt), [upstream license](https://github.com/i2rt-robotics/i2rt/blob/98d177bb511d545c80c0e8ec13ffaf227238a8d6/LICENSE) |
| 6 | i2rt YAM model | Seventeen `model2*` station meshes matching `google-deepmind/mujoco_menagerie@4a7015530bd7a4161103ae8f0905a96481e4cc1a/i2rt_yam` | MIT | [Local license](LICENSES/MIT-i2rt-YAM.txt), [upstream license](https://github.com/google-deepmind/mujoco_menagerie/blob/4a7015530bd7a4161103ae8f0905a96481e4cc1a/i2rt_yam/LICENSE) |
| 7 | Trossen ALOHA D405 model | `d405.stl`, matching `google-deepmind/mujoco_menagerie@4a7015530bd7a4161103ae8f0905a96481e4cc1a/aloha/assets/d405_solid.stl` | BSD-3-Clause | [Local license](LICENSES/BSD-3-Clause-ALOHA.txt), [upstream license](https://github.com/google-deepmind/mujoco_menagerie/blob/4a7015530bd7a4161103ae8f0905a96481e4cc1a/aloha/LICENSE) |
| 8 | Franka Emika Panda model | Sixty-seven meshes and a modified MJCF derived from `google-deepmind/mujoco_menagerie@4a7015530bd7a4161103ae8f0905a96481e4cc1a/franka_emika_panda` | Apache-2.0 | [Upstream license](https://github.com/google-deepmind/mujoco_menagerie/blob/4a7015530bd7a4161103ae8f0905a96481e4cc1a/franka_emika_panda/LICENSE), [local provenance](aspire/sim/cap/envs/assets/franka_pick_place/README.md) |
| 9 | cuRobo v0.7.8 | Dependency patch based on `NVlabs/curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | NVIDIA License; custom/non-OSI, public use limited to noncommercial research or evaluation | [Local license](LICENSES/NVIDIA-cuRobo-v0.7.8.txt), [upstream license](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE) |
| 10 | YAM station XML | `station.xml` and `loose_limit_version/station.xml`, adapted from `uynitsuj/robosuite@97292732ed909ac3ae116579fb768607034a4dbd/robosuite/models/assets/robots/yam/station.xml` | MIT as distributed by the pinned fork, with DeepMind MuJoCo Apache-2.0 attribution; four referenced CAD meshes remain release-gated | [Local license](LICENSES/MIT-Robosuite.txt), [fork license](https://github.com/uynitsuj/robosuite/blob/97292732ed909ac3ae116579fb768607034a4dbd/LICENSE) |

## Pinned Git submodules

The parent repository records only gitlinks. A recursive clone obtains these
repositories from their configured remotes, and their own terms govern their
source, models, datasets, and assets.

| No. | Submodule and exact pin | License | License link |
| ---: | --- | --- | --- |
| 1 | `LIBERO-PRO@47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a` | MIT | [License](https://github.com/uynitsuj/LIBERO-PRO/blob/47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a/LICENSE) |
| 2 | `uynitsuj/robosuite@97292732ed909ac3ae116579fb768607034a4dbd` | MIT, with DeepMind MuJoCo Apache-2.0 attribution | [License](https://github.com/uynitsuj/robosuite/blob/97292732ed909ac3ae116579fb768607034a4dbd/LICENSE) |
| 3 | `Max-Fu/robosuite@a498b087d4bc5a3981e3d27030d09bc537a537f3` | MIT, with DeepMind MuJoCo Apache-2.0 attribution | [License](https://github.com/Max-Fu/robosuite/blob/a498b087d4bc5a3981e3d27030d09bc537a537f3/LICENSE) |
| 4 | `sam3@6fe87d64a5beb9084923d7a9e002741178635b09` | SAM License; custom/non-OSI terms covering redistribution, acknowledgment, trade controls, and prohibited uses | [License](https://github.com/Max-Fu/sam3/blob/6fe87d64a5beb9084923d7a9e002741178635b09/LICENSE) |
| 5 | `curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | NVIDIA License described above, plus component-specific asset licenses | [License](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE), [asset licenses](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE_ASSETS) |
| 6 | `b1k@272ec5ca9936453c4a8fd335c4dfba61245e33ca` | Mixed: OmniGibson, BDDL, and JoyLo use MIT; robot assets include Apache-2.0 material; the Pixar HumanFemale asset permits only personal noncommercial USD testing and prohibits redistribution without written authorization | [OmniGibson](https://github.com/qingh097/b1k/blob/272ec5ca9936453c4a8fd335c4dfba61245e33ca/OmniGibson/LICENSE), [BDDL](https://github.com/qingh097/b1k/blob/272ec5ca9936453c4a8fd335c4dfba61245e33ca/bddl3/LICENSE), [JoyLo](https://github.com/qingh097/b1k/blob/272ec5ca9936453c4a8fd335c4dfba61245e33ca/joylo/LICENSE), [Pixar asset](https://github.com/qingh097/b1k/blob/272ec5ca9936453c4a8fd335c4dfba61245e33ca/asset_pipeline/b1k_pipeline/tools/HumanFemale/LICENSE.txt) |

## External runtime components

These components are not redistributed in the parent repository. Users obtain
them separately, and their vendor or upstream terms apply.

| Component | Distribution status | Terms |
| --- | --- | --- |
| BundleSDF / BundleTrack | Source, checkpoints, and runtime payload are intentionally excluded | No approved redistribution grant has been established for the recovered source snapshot; review the [recovery record](aspire/real/install/locks/bundlesdf/README.md) before use or distribution. |
| AnyGrasp SDK, server, and checkpoint | Vendor source, demo-derived server code, model, credentials, and machine license are intentionally excluded; ASPIRE retains only its client protocol and external-service launcher | Vendor terms apply. The separately installed `graspnetAPI` package is MIT, but that license does not cover the AnyGrasp SDK, demo, server, or model. |
| Contact-GraspNet source and checkpoints | The former recursive submodule is removed; users supply source and checkpoints separately through `CONTACT_GRASPNET_ROOT` and `CONTACT_GRASPNET_CHECKPOINT_DIR` | The custom [NVIDIA Source Code License for Contact-GraspNet](https://github.com/NVlabs/contact_graspnet/blob/master/License.pdf) applies to source. No checkpoint or test-data redistribution right is asserted by ASPIRE. |
| Stereolabs ZED SDK and `pyzed` | Downloaded from Stereolabs during workstation setup | Proprietary [Stereolabs Software and Services License Agreement](https://www.stereolabs.com/legal) applies. |
| SAM 2.1 model assets | Downloaded separately from Hugging Face | Apache-2.0; see the [official model repository](https://huggingface.co/facebook/sam2.1-hiera-large). |
| SAM 3 model assets | Downloaded separately from a gated Hugging Face repository | The custom SAM License and gated-access conditions apply; see the [official model repository](https://huggingface.co/facebook/sam3). |

## Provenance not yet cleared for redistribution

The following tracked material must not be treated as Apache-2.0 or approved
for public redistribution until its source and grant are established. Listing
an item here is not permission to use or distribute it.

| Material | Affected path or record | Open issue |
| --- | --- | --- |
| Remaining YAM CAD | `aspire/real/robot/models/station/assets/base_visual_gate.stl`, `gripper.stl`, `gripper_finger.stl`, and `zed2i.stl` | Written source/owner and redistribution confirmation is pending. Confirm the applicable grant or remove, externalize, or replace these meshes before release. |

The source-file ownership boundary and exact CaP-X/NVIDIA path mapping are
recorded in [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) and the accompanying
[`SOURCE_PROVENANCE.tsv`](SOURCE_PROVENANCE.tsv).
