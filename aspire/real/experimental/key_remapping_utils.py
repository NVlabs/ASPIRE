# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key remapping utilities for observation/action mapping between environment and policy formats."""

from typing import Any, Dict, Literal

import numpy as np
from PIL import Image

from experimental.embodiment_tags import EmbodimentTag

# =============================================================================
# Key maps for observation/action mapping
# =============================================================================

PROPRIO_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

ACTION_KEY_MAP_xdof_oss = {
    "left_joint_pos": "left_joint_pos",
    "left_gripper_pos": "left_gripper_pos",
    "right_joint_pos": "right_joint_pos",
    "right_gripper_pos": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_oss = {
    "top_camera_image": "top",
    "left_camera_image": "left",
    "right_camera_image": "right",
}

PROPRIO_KEY_MAP_xdof = {
    "left_joint_pos": "joint_pos_obs_left",
    "left_gripper_pos": "gripper_pos_obs_left",
    "right_joint_pos": "joint_pos_obs_right",
    "right_gripper_pos": "gripper_pos_obs_right",
}

PROPRIO_KEY_MAP_xdof_ee = {
    "left_ee_pos": "ee_pos_obs_left",
    "left_ee_quat_xyzw": "ee_quat_obs_left",
    "left_gripper_pos": "gripper_pos_obs_left",
    "right_ee_pos": "ee_pos_obs_right",
    "right_ee_quat_xyzw": "ee_quat_obs_right",
    "right_gripper_pos": "gripper_pos_obs_right",
}

ACTION_KEY_MAP_xdof = {
    "joint_pos_action_left": "left_joint_pos",
    "gripper_pos_action_left": "left_gripper_pos",
    "joint_pos_action_right": "right_joint_pos",
    "gripper_pos_action_right": "right_gripper_pos",
}

ACTION_KEY_MAP_xdof_ee = {
    "ee_pos_action_left": "left_ee_pos",
    "ee_quat_action_left": "left_ee_quat_xyzw",
    "gripper_pos_action_left": "left_gripper_pos",
    "ee_pos_action_right": "right_ee_pos",
    "ee_quat_action_right": "right_ee_quat_xyzw",
    "gripper_pos_action_right": "right_gripper_pos",
}

CAMERA_KEY_MAP_xdof_240 = {
    "top_camera_image": "observation.images.top_camera-images-rgb_320_240",
    "left_camera_image": "observation.images.left_camera-images-rgb_320_240",
    "right_camera_image": "observation.images.right_camera-images-rgb_320_240",
}

CAMERA_KEY_MAP_xdof_480 = {
    "top_camera_image": "observation.images.top_camera-images-rgb",
    "left_camera_image": "observation.images.left_camera-images-rgb",
    "right_camera_image": "observation.images.right_camera-images-rgb",
}


# =============================================================================
# Observation/action mapping functions
# =============================================================================


def map_observation(
    observation: Dict[str, Any], embodiment_tag: EmbodimentTag, resolution: Literal[240, 480]
):
    """
    Map environment observation to (images, proprio) format for policy.

    Automatically detects whether observation is in joint space or ee_pose space
    based on available keys.
    """
    proprio = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof_oss
        CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        # Detect observation format: check if ee_pose keys exist
        has_ee_pose = "left_ee_pos" in observation or "right_ee_pos" in observation
        has_joint_pos = "left_joint_pos" in observation or "right_joint_pos" in observation

        if has_ee_pose:
            # Use ee_pose mapping
            PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof_ee
        elif has_joint_pos:
            # Use joint position mapping
            PROPRIO_KEY_MAP = PROPRIO_KEY_MAP_xdof
        else:
            raise ValueError(
                "Observation must contain either joint positions (left_joint_pos, right_joint_pos) "
                "or ee_pose (left_ee_pos, right_ee_pos)"
            )

        if resolution == 240:
            CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_240
        elif resolution == 480:
            CAMERA_KEY_MAP = CAMERA_KEY_MAP_xdof_480
        else:
            raise ValueError(f"Resolution {resolution} not supported")
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")

    for key, value in PROPRIO_KEY_MAP.items():
        if key in observation:
            proprio[value] = observation[key]
    images = {}
    for key, value in CAMERA_KEY_MAP.items():
        if resolution == 240:
            images[value] = Image.fromarray(observation[key]).resize((320, 240))
        elif resolution == 480:
            images[value] = Image.fromarray(observation[key])
        else:
            raise ValueError(f"Resolution {resolution} not supported")
    return images, proprio


def map_action(action: Dict[str, Any], embodiment_tag: EmbodimentTag):
    """
    Map policy action to environment action format.

    Automatically detects whether action is in joint space or ee_pose space
    based on available keys. If action is already in environment format (e.g.,
    from replay policy), it passes through unchanged.
    """
    action_dict = {}
    if embodiment_tag == EmbodimentTag.XDOF_OSS_DATA:
        ACTION_KEY_MAP = ACTION_KEY_MAP_xdof_oss
    elif embodiment_tag == EmbodimentTag.XDOF:
        # Check if action is already in environment format (from replay policy)
        # If it has environment keys directly, pass through unchanged
        env_keys = [
            "left_ee_pos", "left_ee_quat_xyzw", "right_ee_pos", "right_ee_quat_xyzw",
            "left_joint_pos", "right_joint_pos",
            "left_gripper_pos", "right_gripper_pos"
        ]
        if any(key in action for key in env_keys):
            # Action is already in environment format, pass through
            return action

        # Detect action format: check if policy format keys exist
        has_ee_pose = "ee_pos_action_left" in action or "ee_quat_action_left" in action
        has_joint_pos = "joint_pos_action_left" in action

        if has_ee_pose:
            # Use ee_pose mapping
            ACTION_KEY_MAP = ACTION_KEY_MAP_xdof_ee
        elif has_joint_pos:
            # Use joint position mapping
            ACTION_KEY_MAP = ACTION_KEY_MAP_xdof
        else:
            raise ValueError(
                f"Action must contain either joint positions (joint_pos_action_left, joint_pos_action_right) "
                f"or ee_pose (ee_pos_action_left, ee_quat_action_left) or be in environment format "
                f"(left_joint_pos, left_ee_pos, etc.). Got keys: {list(action.keys())}"
            )
    else:
        raise ValueError(f"Embodiment tag {embodiment_tag} not supported")

    for key, value in ACTION_KEY_MAP.items():
        if key in action:
            action_dict[value] = action[key]
    return action_dict


# =============================================================================
# Portal serialization helper
# =============================================================================


def _make_arrays_contiguous(obj: Any, memo: dict[Any, Any] | None = None) -> Any:
    """Recursively ensure all numpy arrays are contiguous for Portal serialization."""
    # Handles loops in the object graph
    if memo is None:
        memo = dict()
    if id(obj) in memo:
        return memo[id(obj)]

    if isinstance(obj, np.ndarray):
        return np.ascontiguousarray(obj)
    elif isinstance(obj, dict):
        new_obj = {}
        memo[id(obj)] = new_obj
        for k, v in obj.items():
            new_obj[k] = _make_arrays_contiguous(v, memo)
        return new_obj
    elif isinstance(obj, (list, tuple)):
        new_obj = []
        memo[id(obj)] = new_obj
        for item in obj:
            new_obj.append(_make_arrays_contiguous(item, memo))
        return new_obj
    return obj


# =============================================================================
# Action utilities
# =============================================================================


def hold_action_from_proprio(proprio: dict[str, Any]) -> dict[str, Any]:
    """Create a hold action from proprioceptive observations.

    Supports both joint space and ee_pose space observations.
    Returns actions in the same space as the observations.
    """
    # Try ee_pose format first (for cartesian_position control mode)
    ee_pose_keys = [
        ("ee_pos_obs_left", "ee_quat_obs_left", "gripper_pos_obs_left",
         "ee_pos_obs_right", "ee_quat_obs_right", "gripper_pos_obs_right"),
    ]
    for lep, leq, lg, rep, req, rg in ee_pose_keys:
        if lep in proprio and rep in proprio:
            return {
                "left_ee_pos": np.asarray(proprio[lep], dtype=np.float32).reshape(-1)[:3],
                "left_ee_quat_xyzw": np.asarray(
                    proprio.get(leq, np.array([0, 0, 0, 1], dtype=np.float32)), dtype=np.float32
                ).reshape(-1)[:4],
                "left_gripper_pos": np.asarray(
                    proprio.get(lg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "right_ee_pos": np.asarray(proprio[rep], dtype=np.float32).reshape(-1)[:3],
                "right_ee_quat_xyzw": np.asarray(
                    proprio.get(req, np.array([0, 0, 0, 1], dtype=np.float32)), dtype=np.float32
                ).reshape(-1)[:4],
                "right_gripper_pos": np.asarray(
                    proprio.get(rg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "source": None,
            }

    # Try joint space format (for joint_position control mode)
    joint_keys = [
        ("left_joint_pos", "left_gripper_pos", "right_joint_pos", "right_gripper_pos"),
        (
            "joint_pos_obs_left",
            "gripper_pos_obs_left",
            "joint_pos_obs_right",
            "gripper_pos_obs_right",
        ),
    ]
    for lq, lg, rq, rg in joint_keys:
        if lq in proprio and rq in proprio:
            return {
                "left_joint_pos": np.asarray(proprio[lq], dtype=np.float32).reshape(-1)[:6],
                "left_gripper_pos": np.asarray(
                    proprio.get(lg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "right_joint_pos": np.asarray(proprio[rq], dtype=np.float32).reshape(-1)[:6],
                "right_gripper_pos": np.asarray(
                    proprio.get(rg, np.zeros(1, np.float32)), dtype=np.float32
                ).reshape(-1)[:1],
                "source": None,
            }
    raise ValueError("No valid keys found in proprio")
