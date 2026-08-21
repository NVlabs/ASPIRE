---
name: libero-fix-loop-task-exploration
description: Minimal initial observation step for LIBERO Fix Loop tasks. Captures authoritative task language and two camera views while treating all inferred geometry and strategy as provisional.
---

# Initial Scene Analysis

Capture development seed 51:

```bash
TASK_DIR="outputs/libero_fix_loop/$SUITE/$TASK"
mkdir -p "$TASK_DIR"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
SNAPSHOT_DIR="$TASK_DIR" \
.venv-libero/bin/python3 scripts/libero/replay_trial.py \
  --args.suite "$SUITE" --args.task "$TASK" --args.trial 51 \
  --args.replay-code scripts/libero/scene_snapshot.py \
  --args.config env_configs/libero/franka_libero_traced.yaml \
  --args.output-dir outputs/libero_fix_loop_exploration
```

Read `scene_snapshot.jpg`, `scene_snapshot_wrist.jpg`, and the printed `TASK_LANGUAGE`. For `_task` suites, task language overrides the filename.

Write only a short `task_analysis.md`:

```markdown
# Initial Task Analysis

- Actual task language:
- Manipulated object and likely prompts:
- Goal/handle and likely prompts:
- Visible obstacles and plausible approach:
- Interaction type: pick/place | push/contact | articulated | multi-stage
- Uncertainties to verify from later seeds:
```

This analysis comes from one scene and may be flawed. Treat object identity, relative position, dimensions, free space, and motion strategy as hypotheses. Do not encode snapshot-specific pixel ordering or coordinates. Revise the analysis when multi-seed traces or keyframes disagree.
