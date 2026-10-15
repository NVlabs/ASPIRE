"""First-class Robosuite task-environment definitions.

Custom MuJoCo environment subclasses that define registered Robosuite tasks (not
validation or example scaffolding). Currently the reverse-stack ``CubeRestack``
specialization backing the ``cube_restack`` task.
"""

from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.stack import Stack


class CubeRestack(Stack):
    """Start cube A on B and succeed only after cube B is stacked on A."""

    def _reset_internal(self) -> None:
        super()._reset_internal()
        if self.deterministic_reset:
            return
        cube_b = np.asarray(self.sim.data.get_joint_qpos(self.cubeB.joints[0]), dtype=float)
        cube_a = np.asarray(self.sim.data.get_joint_qpos(self.cubeA.joints[0]), dtype=float)
        cube_a[:2] = cube_b[:2]
        cube_a[2] = cube_b[2] + self.cubeB.top_offset[2] - self.cubeA.bottom_offset[2]
        self.sim.data.set_joint_qpos(self.cubeA.joints[0], cube_a)

    def _check_success(self) -> bool:
        cube_a = self.sim.data.body_xpos[self.cubeA_body_id]
        cube_b = self.sim.data.body_xpos[self.cubeB_body_id]
        horizontal_error = float(np.linalg.norm(cube_a[:2] - cube_b[:2]))
        b_above_a = cube_b[2] > cube_a[2] + 0.02
        touching = self.check_contact(self.cubeA, self.cubeB)
        grasping_b = self._check_grasp(self.robots[0].gripper, self.cubeB)
        return bool(horizontal_error < 0.02 and b_above_a and touching and not grasping_b)

    def reward(self, action=None) -> float:
        del action
        return float(self._check_success())


__all__ = ["CubeRestack"]
