# LIBERO Suite Reference

This page is the compact LIBERO/LIBERO-Pro reference for ASPIRE. Use it to identify task suites, environment conventions, and the correct public experiment entrypoints.

The setup path is:

1. Complete the base repository setup in the root [README.md](../README.md).
2. Complete the root README's "Continuation: LIBERO-PRO" setup.
3. Follow the suite guide at [`.claude/libero/CLAUDE.md`](../.claude/libero/CLAUDE.md).
4. Start from the relevant experiment `INSTRUCTIONS.md` file under [`.claude/libero/`](../.claude/libero/).

LIBERO paper reproduction does not require a local LLM proxy. Pass model/API credentials to your CLI agent or to the specific batch runner command described in [run-baseline.md](../.claude/libero/run-baseline.md).

## Environment Conventions

LIBERO-PRO uses a dedicated virtual environment because its Robosuite dependency conflicts with the standalone Robosuite stack:

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

Use `.venv-libero/bin/python3` for LIBERO replay/eval runners and perception servers. Use `.venv/bin/python3` only for lightweight progress, plotting, and analysis scripts.

For headless replay/eval:

```bash
export ASPIRE_ROOT="$(pwd)"
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Create the LIBERO path config as shown in the root README. It should point to the checked-out `cap/third_party/LIBERO-PRO` submodule paths. This user-level config is required by upstream LIBERO; without it, imports such as `from libero import benchmark` can prompt for paths interactively and fail in non-interactive runs.

## Perception Servers

LIBERO replay/eval uses three local perception/IK servers:

| Server | Port | Launcher |
|---|---:|---|
| SAM3 | 8114 | `cap/serving/launch_sam3_server.py` |
| GraspNet | 8115 | `cap/serving/launch_contact_graspnet_server.py` |
| PyRoKi | 8116 | `cap/serving/launch_pyroki_server.py` |

The experiment coordinator normally checks these ports and starts missing servers during preflight. Manual prelaunch is optional, but must run in a persistent terminal such as tmux:

```bash
tmux new -s aspire-perception
cd "$ASPIRE_ROOT"
export ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3
bash scripts/common/start_perception_servers.sh --no-molmo

for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

SAM3 uses gated Hugging Face weights. Authenticate before starting servers, and do not store tokens in the repository. `404` on `/health` means these FastAPI processes are alive; `000` means down.

## Experiment Entrypoints

The public LIBERO experiments are agent-driven. A coordinator agent reads the relevant guide, runs preflight checks, dispatches per-task subagents, and uses replay/eval scripts for scoring.

| Experiment | Entry point |
|---|---|
| Fix Loop | [`.claude/libero/fix-loop/INSTRUCTIONS.md`](../.claude/libero/fix-loop/INSTRUCTIONS.md) |
| Fix Loop + Evolutionary Search | [`.claude/libero/evosearch/INSTRUCTIONS.md`](../.claude/libero/evosearch/INSTRUCTIONS.md) |
| Zero-Shot Transfer | [`.claude/libero/zeroshot-transfer/INSTRUCTIONS.md`](../.claude/libero/zeroshot-transfer/INSTRUCTIONS.md) |
| Library-Size Scaling | [`.claude/libero/library-size-scaling/INSTRUCTIONS.md`](../.claude/libero/library-size-scaling/INSTRUCTIONS.md) |
| Inference-Time Scaling | [`.claude/libero/inference-time-scaling/INSTRUCTIONS.md`](../.claude/libero/inference-time-scaling/INSTRUCTIONS.md) |
| Baseline collection | [`.claude/libero/run-baseline.md`](../.claude/libero/run-baseline.md) |

## Suite Families

| Suite family | Role |
|---|---|
| `libero_goal_swap`, `libero_goal_task`, `libero_object_swap`, `libero_object_task`, `libero_spatial_swap`, `libero_spatial_task` | LIBERO-Pro Fix Loop and Evolutionary Search experiments |
| `libero_90` | Skill-library build set for zero-shot transfer and scaling |
| `libero_10_swap`, `libero_10_task` | LIBERO-Long-Pro eval set for zero-shot and scaling experiments |
| `libero_10`, `libero_90`, `libero_object`, `libero_spatial`, `libero_goal` | Base LIBERO suites used for reference and development |

Critical suite rule: `_task` variants remap the language instruction. Always read the task language from the environment or benchmark API; do not infer the goal from the BDDL filename.

To list suites and tasks from the installed LIBERO package:

```bash
source .venv-libero/bin/activate
python - <<'PY'
from libero import benchmark

benchmarks = benchmark.get_benchmark_dict()
for suite_name in sorted(benchmarks):
    suite = benchmarks[suite_name]()
    print(f"\n{suite_name}: {suite.n_tasks} tasks")
    for i in range(suite.n_tasks):
        print(f"  [{i}] {suite.get_task(i).language}")
PY
```

## Config Pattern

LIBERO task configs select a suite and task index under `low_level`:

```yaml
low_level:
  _target_: aspire.sim.cap.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 2
```

The task prompt should use `{libero_environment_goal}` so the runtime task language is inserted from LIBERO.

Common configs:

| Config | Use |
|---|---|
| `env_configs/libero/franka_libero_traced.yaml` | Replay/eval with trace logging |
| `env_configs/libero/franka_libero_baseline_debug.yaml` | Baseline debug with text and trace feedback |
| `env_configs/libero/franka_libero_baseline_image_diff_debug.yaml` | Baseline debug with image-difference feedback |
| `env_configs/libero/franka_libero_libero10_traced.yaml` | LIBERO-10 replay/eval |

For a single replay of generated code:

```bash
MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite <suite_name> \
  --args.task "<task language>" \
  --args.trial <seed> \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.replay-code /path/to/code.py
```

For baseline sweeps, use [`.claude/libero/run-baseline.md`](../.claude/libero/run-baseline.md).
