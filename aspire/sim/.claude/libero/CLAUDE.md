# LIBERO Suite Guide

This is the suite-level reference for all LIBERO experiments in ASPIRE. Use it for benchmark roles, shared APIs, environment conventions, and links to the five experiment entrypoints.

## Experiment Entrypoints

| Experiment | Entry point | Pipeline reference |
|---|---|---|
| Fix Loop | [fix-loop/INSTRUCTIONS.md](fix-loop/INSTRUCTIONS.md) | [fix-loop/SKILL.md](fix-loop/SKILL.md) |
| Fix Loop + Evolutionary Search | [evosearch/INSTRUCTIONS.md](evosearch/INSTRUCTIONS.md) | [evosearch/SKILL.md](evosearch/SKILL.md) |
| Zero-Shot Transfer | [zeroshot-transfer/INSTRUCTIONS.md](zeroshot-transfer/INSTRUCTIONS.md) | [zeroshot-transfer/SKILL.md](zeroshot-transfer/SKILL.md) |
| Library-Size Scaling | [library-size-scaling/INSTRUCTIONS.md](library-size-scaling/INSTRUCTIONS.md) | [library-size-scaling/SKILL.md](library-size-scaling/SKILL.md) |
| Inference-Time Scaling | [inference-time-scaling/INSTRUCTIONS.md](inference-time-scaling/INSTRUCTIONS.md) | [inference-time-scaling/SKILL.md](inference-time-scaling/SKILL.md) |

Human-facing setup starts at the root `README.md`; experiment execution starts from the `INSTRUCTIONS.md` files below.

## Suite-Shared Files

| File | Purpose |
|---|---|
| [api-reference.md](api-reference.md) | LIBERO reduced API functions, output structure, TraceLogger format, source files |
| [run-baseline.md](run-baseline.md) | Baseline collection reference |
| [skills/](skills/) | Suite-shared, learned robot-control knowledge |
| [analysis/](analysis/) | Token accounting and other non-policy analysis procedures |
| [legacy/](legacy/) | Older pipeline and ablation references, not public experiment entrypoints |

The public runbooks here keep suite-shared clean templates. Frozen learned skill snapshots are maintained separately on branch `learned-skills` under the same `.claude/libero/<experiment>/skills/` layout.

## Benchmark Roles

| Suite family | Role |
|---|---|
| `libero_goal_swap`, `libero_goal_task`, `libero_object_swap`, `libero_object_task`, `libero_spatial_swap`, `libero_spatial_task` | LIBERO-Pro Fix Loop and Evolutionary Search experiments |
| `libero_90` | Skill-library build set for zero-shot transfer and scaling |
| `libero_10_swap`, `libero_10_task` | LIBERO-Long-Pro eval set for zero-shot and scaling experiments |

Critical suite rule: `_task` variants remap the language instruction. Always read `env.handle.task_language`; do not trust the BDDL filename as the task goal.

## Environment Conventions

Use `.venv-libero/bin/python3` for LIBERO eval runners and perception servers. Use `.venv/bin/python3` only for lightweight progress, analysis, and plotting scripts.

The LIBERO environment must be synced with both `--extra libero` and
`--extra contactgraspnet`, and `~/.libero/config.yaml` must point to the
checked-out `cap/third_party/LIBERO-PRO` paths. Upstream LIBERO reads this
user-level config at import time; without it, non-interactive agents can fail on
an interactive path prompt.

Required for replay/eval:

```bash
export ASPIRE_ROOT="$(pwd)"
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Perception servers must be up before replay/eval. SAM3 requires authenticated
access to the gated Hugging Face `facebook/sam3` model. Start servers in a
persistent terminal such as tmux; do not launch them from a one-off shell that
will exit immediately after startup.

```bash
need_servers=false
for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health 2>/dev/null || echo 000)
  echo "port $p: $code"
  [[ "$code" == "000" ]] && need_servers=true
done

if $need_servers; then
  tmux new -d -s aspire-perception \
    "cd '$ASPIRE_ROOT' && ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3 bash scripts/common/start_perception_servers.sh --no-molmo; sleep infinity"
fi
```

## Critical Rules

1. Never use forbidden simulator ground-truth APIs; see root `CLAUDE.md`.
2. Keep build/debug seeds separate from held-out eval seeds.
3. Subagents write task artifacts and findings; coordinators update shared skills.
4. Store reusable robot-control patterns in [skills/](skills/), not inside transient output folders.
5. Log significant experiment work to `docs/logs/YYYY-MM-DD.md`.
