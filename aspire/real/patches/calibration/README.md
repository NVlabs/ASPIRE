# Calibration Recovery

`station-xml-local.patch` contains exactly the three local line changes against
`jia-xie/yam-calibration` commit
`fd70272c0ece327f636331ace5e30a1738f7ca65`. The changes update the left wrist,
right wrist, and top D405 camera extrinsics. The calibrated XML itself is not
included.

Apply from the calibration repository root with:

```bash
git apply /path/to/station-xml-local.patch
```

The active XML uses `meshdir="assets/"` and references the following meshes:

| Mesh name | File | Bytes | SHA-256 | Validation |
| --- | --- | ---: | --- | --- |
| base_visual_gate | base_visual_gate.stl | 9126284 | `7455ecaa1a22959ccb01fc5d0ef2f4c7de04a46648611be716c4d45c9b9259bf` | valid binary STL |
| camera_d405 | d405.stl | 1242284 | `8a9a84dd9c9a67687e5ad0b37a7ed334dbfea6b0f995fef28524cc8da5f21ed2` | valid binary STL |
| model2 | model2.stl | 524284 | `5ac61c0313ed00c38655e41681af1b8b307cf8d5ef57209a75f215cea02df754` | valid binary STL |
| model2__2 | model2__2.stl | 524284 | `30dab41f14bae95d737abca3c934eb5d570ab107dff0b40952fbdfb9d07eab79` | valid binary STL |
| model2__3 | model2__3.stl | 80684 | `9fd96f5ad387fe7dd96eaa65e386637800e2f5637b6167fbad4ecb37d8598f47` | valid binary STL |
| model2__4 | model2__4.stl | 80684 | `33c0e7ef538e2c4473efa9c4ef2a8441b6094b65090b4dec71abe82001d974df` | valid binary STL |
| model2__5 | model2__5.stl | 524284 | `09f08e7dead2d6363618471adadb1b94af39e8056676670f3fd21ec06c197b20` | valid binary STL |
| model2__6 | model2__6.stl | 86284 | `888b5919d10dbfb59b964c3cafc2ddbdac3b5426e20dd13bcaf4baedd5ee15c9` | valid binary STL |
| model2__7 | model2__7.stl | 52084 | `2d64b18f1e70b0460eb53335d340b7c879976a3093ec67b07bcb7245e371a91d` | valid binary STL |
| model2__8 | model2__8.stl | 524284 | `db3708da6b14daf3fc9a1909a8325b68f65ff69146cdd20b69a9ddc4ce51f50d` | valid binary STL |
| model2__9 | model2__9.stl | 52084 | `74969d09b8b7f75cc984c5c7e5526d2941c6724f48b3efa1fa8730cc7820f4f5` | valid binary STL |
| model2__10 | model2__10.stl | 524284 | `b879cc07f547b7a2ac52e428dd206e80a0d4b0e805cb56b9a8230df04a690923` | valid binary STL |
| model2__11 | model2__11.stl | 248584 | `af220d2a51e8444392aef36e2fe4a6f69aed22bc292a7f38e54da3b4f5253434` | valid binary STL |
| model2__12 | model2__12.stl | 68084 | `e79614869bd2283f70fafa150ce748d5b6c03377a651448c6fe9893a22f6d1de` | valid binary STL |
| model2__13 | model2__13.stl | 524284 | `32b62f14f00ea82bfc79b935f7dac32c7cb276a3abc0aae8362cbba3fb1326de` | valid binary STL |
| model2__14 | model2__14.stl | 524284 | `72d2178789dd5e0358c2caf9dad90b8881e505f4c52a8d6d18c71b5e93e8c84b` | valid binary STL |
| model2__15 | model2__15.stl | 183684 | `ca8788184d54f30c18482cb9b48dffb010a93dbc029054b6cc515083a740d324` | valid binary STL |
| model2__16 | model2__16.stl | 183684 | `18bb99691eed524393e8f3a0ff29e79d97ab75689bc92144bbdfafa2cc79d37b` | valid binary STL |
| model2__17 | model2__17.stl | 524284 | `6513dbd192d2bf3a4f7016231f1ba5eca93916c9ebfdc334e9b435ca1d302ff6` | valid binary STL |

Validation checks the binary STL triangle count against file length. The file
utility mislabels several binary STL headers as RenderWare collision data, so
its MIME guess was not used as the validity test.

`model2__1.stl` exists in the same asset directory but is not referenced by
the active XML. It is a 10,532-byte HTML document with SHA-256
`ba76b2888630a6e76c98e5791d67328441515ce8652b6f5ee377a3f528132efa`
and must not be treated as geometry. Remove it before public release.

## Redistribution status

The following mesh sources and licenses were independently verified:

- `model2.stl` and `model2__2.stl` through `model2__17.stl` are byte-identical
  to the `i2rt_yam` model in `google-deepmind/mujoco_menagerie` commit
  `4a7015530bd7a4161103ae8f0905a96481e4cc1a`, which is MIT-licensed; see
  `LICENSES/MIT-i2rt-YAM.txt` at the repository root.
- `d405.stl` is byte-identical to `aloha/assets/d405_solid.stl` at the same
  MuJoCo Menagerie commit. The ALOHA model is BSD-3-Clause licensed by Trossen
  Robotics; see `LICENSES/BSD-3-Clause-ALOHA.txt` at the repository root.

The calibration repository is `https://github.com/jia-xie/yam-calibration.git`
and has no LICENSE file. `base_visual_gate.stl` also appears in the pinned
Robosuite fork, but no directory-specific source or redistribution grant was
found. No written owner confirmation was found for that mesh or for the
calibration XML and camera-extrinsic changes. Public release of those
unresolved items remains blocked until the owner confirms all of the following
in writing:

- whether the generated camera extrinsics may be published;
- whether `base_visual_gate.stl` may be redistributed and under which license;
- whether the station geometry contains vendor-proprietary CAD;
- the required attribution and notices;
- whether modified/derived mesh files may be distributed.

This document records the missing confirmation; it is not itself permission.
