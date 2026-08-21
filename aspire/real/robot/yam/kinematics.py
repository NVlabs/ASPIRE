# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import mink
import mujoco
import numpy as np


class YamKinematics:
    def __init__(
        self,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        lm_damping: float = 1.0,
    ):
        from robot.models.station.paths import get_station_xml
        model_path = get_station_xml()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        self.configuration = mink.Configuration(model)
        self.tasks = [
            mink.FrameTask(
                frame_name=f"{side}_grasp_site",
                frame_type="site",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=lm_damping,
            )
            for side in ["left", "right"]
        ]
        self.left_end_effector_task, self.right_end_effector_task = self.tasks

    def forward_kinematics(
        self, left_joint_pos: np.ndarray, right_joint_pos: np.ndarray
    ) -> np.ndarray:
        # Update qpos
        self.configuration.data.qpos[:6] = left_joint_pos
        self.configuration.data.qpos[8:14] = right_joint_pos
        self.configuration.update()

        # Compute forward kinematics
        left_ee_pose = self.configuration.get_transform_frame_to_world("left_grasp_site", "site")
        right_ee_pose = self.configuration.get_transform_frame_to_world("right_grasp_site", "site")

        # Extract end effector pos and quat
        left_ee_pos = left_ee_pose.translation()
        left_ee_quat_xyzw = left_ee_pose.rotation().wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw
        right_ee_pos = right_ee_pose.translation()
        right_ee_quat_xyzw = right_ee_pose.rotation().wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw

        return left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw

    def inverse_kinematics(
        self,
        left_ee_pos: np.ndarray,
        left_ee_quat_xyzw: np.ndarray,
        right_ee_pos: np.ndarray,
        right_ee_quat_xyzw: np.ndarray,
        seeded=False,
        dt=0.01,
        solver="daqp",
        damping=1e-3,
        err_threshold=1e-4,
        max_iters=20,
    ) -> np.ndarray:
        left_ee_quat = left_ee_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
        right_ee_quat = right_ee_quat_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz

        # Initialize qpos to zero if not seeding from previous joint configuration
        if not seeded:
            self.configuration.update(np.zeros_like(self.configuration.data.qpos))

        # Update targets
        T_wt_left = mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=left_ee_quat), left_ee_pos)
        T_wt_right = mink.SE3.from_rotation_and_translation(
            mink.SO3(wxyz=right_ee_quat), right_ee_pos
        )
        self.left_end_effector_task.set_target(T_wt_left)
        self.right_end_effector_task.set_target(T_wt_right)

        # Solve IK
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, dt, solver, damping)
            self.configuration.integrate_inplace(vel, dt)
            err_left = self.left_end_effector_task.compute_error(self.configuration)
            err_right = self.right_end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err_left) <= err_threshold
                and np.linalg.norm(err_right) <= err_threshold
            ):
                break

        # Extract joint positions
        qpos = self.configuration.q
        left_joint_pos = qpos[:6]
        right_joint_pos = qpos[8:14]

        return left_joint_pos, right_joint_pos


def main():
    import time

    kinematics = YamKinematics()
    left_joint_pos = np.array([-0.3, 1.35, 1.6, -0.8, 0.3, -0.25])
    right_joint_pos = np.array([0.3, 1.35, 1.6, -0.8, -0.3, 0.25])

    # Forward kinematics
    left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw = (
        kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
    )

    # Inverse kinematics
    left_joint_pos_ik, right_joint_pos_ik = kinematics.inverse_kinematics(
        left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw
    )

    # Check consistency
    np.testing.assert_allclose(left_joint_pos, left_joint_pos_ik, atol=1e-3)
    np.testing.assert_allclose(right_joint_pos, right_joint_pos_ik, atol=1e-3)

    # Benchmark FK
    num_iters = 1000
    start_time = time.time()
    for _ in range(num_iters):
        # 0.015 ms (gear-ax8-max-05)
        kinematics.forward_kinematics(left_joint_pos, right_joint_pos)
    end_time = time.time()
    print(f"FK time: {1000 * (end_time - start_time) / num_iters:.3f} ms")

    # Benchmark IK
    start_time = time.time()
    for _ in range(num_iters):
        # 4.7 ms starting from zero joint configuration (gear-ax8-max-05)
        # kinematics.inverse_kinematics(left_ee_pos, left_ee_quat_xyzw, right_ee_pos, right_ee_quat_xyzw)

        # 0.37 ms starting from previous joint configuration (gear-ax8-max-05)
        kinematics.inverse_kinematics(
            left_ee_pos,
            left_ee_quat_xyzw,
            right_ee_pos,
            right_ee_quat_xyzw,
            seeded=True,
        )
    end_time = time.time()
    print(f"IK time: {1000 * (end_time - start_time) / num_iters:.3f} ms")


if __name__ == "__main__":
    main()
