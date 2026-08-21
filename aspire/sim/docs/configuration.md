# Configuration Reference

## CLI flags

Override any YAML config field from the command line:

```bash
uv run --no-sync --active python -m aspire.sim.cap.envs.launch \
    --config-path <config.yaml> \
    --model google/gemini-3.1-pro-preview \
    --server-url https://example.com/v1/chat/completions \
    --temperature 1.0 \
    --total-trials 100 \
    --num-workers 12 \
    --record-video True
```

| Flag                | Default                                  | Description                           |
| ------------------- | ---------------------------------------- | ------------------------------------- |
| `--config-path`     | *(required)*                             | Path to YAML task config              |
| `--model`           | `google/gemini-3.1-pro-preview`          | Model name                            |
| `--server-url`      | from YAML/runner                         | OpenAI-compatible LLM endpoint        |
| `--temperature`     | `1.0`                                    | Sampling temperature                  |
| `--total-trials`    | from YAML                                | Number of evaluation trials           |
| `--num-workers`     | from YAML                                | Parallel worker count                 |
| `--use-oracle-code` | `False`                                  | Run human-written reference solutions |

## YAML config format

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: aspire.sim.cap.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: aspire.sim.cap.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12
```

The `_target_` keys enable Hydra-style lazy instantiation via `aspire.sim.cap.envs.configs.instantiate()`.

### Perception servers

LIBERO public experiment coordinators check/start perception servers with `scripts/common/start_perception_servers.sh`. Some YAML configs can also include an `api_servers` section that auto-launches perception servers when the evaluation starts:

```yaml
api_servers:
  - _target_: aspire.sim.cap.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1

  - _target_: aspire.sim.cap.serving.launch_contact_graspnet_server.main
    port: 8115
    host: 127.0.0.1

  - _target_: aspire.sim.cap.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand
```

The launcher automatically:
- Skips servers whose port is already in use (e.g. started externally)
- Waits for all servers to be ready before running trials
- Terminates all servers on exit

If you prefer to manage servers separately (e.g. for sharing across multiple eval runs), use `launch_servers.py`:

```bash
uv run --no-sync --active cap/serving/launch_servers.py --profile default
```

| Profile | Servers | GPU Required |
|---------|---------|-------------|
| `default` | SAM3 (8114) + ContactGraspNet (8115) + PyRoKi (8116) | Yes (~5 GB VRAM) |
| `full` | default + OWL-ViT (8118) + SAM2 (8113) | Yes (~14 GB VRAM) |
| `minimal` | PyRoKi (8116) only | No (CPU-only) |

## Model Providers

ASPIRE generation/eval paths accept OpenAI-compatible chat-completions endpoints through `--server-url` and the corresponding API key flags. The public LIBERO paper reproduction path passes credentials directly to the CLI agent or to the batch runner documented in [run-baseline.md](../.claude/libero/run-baseline.md); it does not require a local LLM proxy.

For local models, run an OpenAI-compatible server such as vLLM and point `--server-url` at it:

```bash
uv run python -m aspire.sim.cap.serving.vllm_server --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8080 --tensor-parallel-size 4
```

Custom provider implementations live under `cap/serving/providers/` and implement a simple `generate_code` method.

Legacy local proxy utilities remain under `cap/serving/` for private workflows, but they are not part of the public LIBERO reproduction setup.
