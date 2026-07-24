<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# YAM Station Assets

The following retained meshes are NVIDIA-owned assets developed internally by
the team and are distributed under the repository's Apache-2.0 license:

- `base_visual_gate.stl`
- `gripper.stl`
- `gripper_finger.stl`, exported from the NVIDIA-owned Fello gripper

## ZED 2i reference geometry

ASPIRE does not redistribute the Stereolabs ZED 2i vendor CAD. The ZED station
XML and URDF use an NVIDIA-authored box proxy matching the vendor model's
axis-aligned bounds while preserving the camera pose, optical frame, and
calibration.

The detailed vendor CAD is available for reference from the
[official Stereolabs 3D-model page](https://www.stereolabs.com/3dmodels) and is
governed by Stereolabs' terms when obtained separately.
