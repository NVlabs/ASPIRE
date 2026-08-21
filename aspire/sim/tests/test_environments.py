# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# uv run python tests/test_environments.py # needs to have reward 1.0

import os 
os.environ.setdefault("MUJOCO_GL", "egl")

from aspire.sim.cap.envs.tasks import CodeExecEnvConfig, CodeExecutionEnvBase, get_exec_env, list_exec_envs, get_config, list_configs
from aspire.sim.cap.envs.base import list_envs
from aspire.sim.cap.integrations.base_api import list_apis
import pytest
import tyro
import time

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1",
        reason="Set ASPIRE_INTEGRATION_REAL=1 and start the IK/perception servers to run simulator execution tests",
    ),
]


def check_environment(
    env_name: str = "franka_pick_place_code_env",
) -> bool:

    print("Available environments: ", list_envs())
    print("Available execution environments: ", list_exec_envs())
    print("Available APIs: ", list_apis())
    print("Available configurations: ", list_configs())
    try:
        cfg = get_config(env_name)
    except KeyError as exc:
        pytest.skip(str(exc))
    if cfg.low_level not in list_envs():
        pytest.skip(f"Low-level environment '{cfg.low_level}' is not registered")
    missing_apis = [api for api in cfg.apis if api not in list_apis()]
    if missing_apis:
        pytest.skip(f"APIs are not registered: {missing_apis}")
    # should be the same as 
    # cfg = CodeExecEnvConfig(
    #     low_level="franka_cubes_low_level",
    #     apis=["FrankaControlApi"],
    # )
    env : CodeExecutionEnvBase = get_exec_env(env_name)(cfg)
    env.enable_video_capture(True)
    start = time.time()
    obs, info = env.reset()
    end = time.time()
    print("Time taken to reset: ", end - start)
    print("Observation keys: ", list(obs.keys()))
    print("Prompt: ", obs["full_prompt"][1]["content"])
    start = time.time()
    obs_next, reward, terminated, truncated, info_step = env.step(env.oracle_code)
    end = time.time()
    print("Time taken: ", end - start)
    # package the video frames into a video
    video_frames = env.get_video_frames()
    if video_frames:
        import imageio
        imageio.mimsave("test_video.mp4", video_frames, fps=30)
        print("Video saved to test_video.mp4")
    if reward != 1.0:     
        # print("Observation: ", obs_next)
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info_step)
        return False
    else:
        print("Success")
        return True

def test_franka_pick_place_code_env() -> None:
    assert check_environment("franka_pick_place_code_env")

def test_franka_robosuite_pick_place_code_env() -> None:
    assert check_environment("franka_robosuite_pick_place_code_env")

def test_franka_lift_code_env() -> None:
    assert check_environment("franka_lift_code_env")
    
def test_franka_nut_assembly_code_env() -> None:
    assert check_environment("franka_nut_assembly_code_env")
    
def test_franka_pick_place_multi_code_env() -> None:
    assert check_environment("franka_pick_place_multi_code_env")
    
def test_r1pro_radio_code_env() -> None:
    assert check_environment("r1pro_radio_code_env")

def test_franka_libero_pick_place_code_env() -> None:
    assert check_environment("franka_libero_pick_place_code_env")

def test_franka_libero_pick_place_code_env_privileged() -> None:
    assert check_environment("franka_libero_pick_place_code_env_privileged")

def test_franka_libero_open_microwave_code_env() -> None:
    assert check_environment("franka_libero_open_microwave_code_env")

def test_franka_libero_open_microwave_code_env_privileged() -> None:
    assert check_environment("franka_libero_open_microwave_code_env_privileged")

if __name__ == "__main__":
    tyro.cli(check_environment)
