import os

import pytest


@pytest.mark.integration
def test_libero_offscreen_env_smoke() -> None:
    if os.environ.get("ASPIRE_INTEGRATION_REAL", "0") != "1":
        pytest.skip("Set ASPIRE_INTEGRATION_REAL=1 to run LIBERO smoke test")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    benchmark = pytest.importorskip("libero.benchmark")
    envs = pytest.importorskip("libero.envs")
    utils = pytest.importorskip("libero.utils")

    os.environ.setdefault("MUJOCO_GL", "egl")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"
    task_suite = benchmark_dict[task_suite_name]()

    task_id = 0
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        utils.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = envs.OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    dummy_action = [0.0] * 7
    for _ in range(10):
        env.step(dummy_action)
    env.close()
