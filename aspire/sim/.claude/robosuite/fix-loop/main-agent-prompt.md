---
name: robosuite/fix-loop/main-agent-prompt
description: Coordinator guide for the Robosuite fix loop. Debug baseline failed trials (seeds 101–125), produce fix_code.py. User runs seeds 1–100 manually. Assigns GPUs round-robin; multiple subagents may share a GPU.
---

# Fix Loop — Robosuite Coordinator Agent Guide

> **What:** Debug baseline failed trials (seeds 101–125), produce fix_code.py. Coordinator runs seeds 1–100 manually.  
> **Baseline:** `outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/` — 7 tasks, seeds 101–125 collected.  
> **Progress:** `docs/progress/fix_loop_robosuite_progress.md` — single source of truth.  
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

---

## Initialization: Verify Perception Servers

Before dispatching any subagents, confirm all four servers are up (404 = UP, 000 = DOWN):

```bash
for p in 8114 8115 8116 8122; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

If any required server is down, start the shared perception servers from a persistent tmux session before dispatching:

```bash
tmux new -s aspire-perception
cd "$ASPIRE_ROOT"
ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3 bash scripts/common/start_perception_servers.sh --no-molmo
```

SAM3 needs authenticated access to the gated Hugging Face `facebook/sam3` model.
GraspNet requires the perception environment to be installed with
`--extra contactgraspnet`. Molmo on `8122` is optional fallback; if it is not
started, point-prompt fallback will be unavailable.

---

## Initialization: Generate Progress File

Before the first dispatch, generate the progress file from disk state:

```bash
.venv-robosuite/bin/python3 scripts/robosuite/gen_progress_robosuite.py
cat docs/progress/fix_loop_robosuite_progress.md
```

---

## The Loop

```
read progress → assign GPUs round-robin (3–7) → dispatch all pending subagents → GO IDLE
                                                                                      ↑
on notification: update skills + GO IDLE (redispatch if crashed) ─────────────────────┘
```

One task = one subagent. GPUs assigned round-robin; multiple subagents may share a GPU.

---

## Coordinator Rules

1. **Dispatch subagents — never debug yourself.** The coordinator's only job is dispatch → idle → collect → update skills → dispatch.
2. **Go idle after dispatching.** You will be notified when a subagent finishes. Do not poll.
3. **Dispatch all pending tasks immediately.** Assign GPUs round-robin (3,4,5,6,7,3,4,...). Multiple tasks may share a GPU.
4. **Never read task-specific debug files.** Do not open `trace.json`, `code.py`, `summary.txt`. You may read `findings.md` and `reasoning.txt` when reviewing completed tasks or writing skill updates.
5. **NEVER re-dispatch a `done` task.** `done` means fix_code.py is written. The user runs seeds 1–100 manually.

---

## Workflow

### 1. Read progress

```bash
cat docs/progress/fix_loop_robosuite_progress.md
```

Find `pending` tasks. (`done` = fix_code.py written — do NOT re-dispatch.)

### 2. Assign GPUs round-robin

GPUs 3–7 are available for MuJoCo workers. Assign tasks to GPUs round-robin (task 1 → GPU 3, task 2 → GPU 4, ..., task 6 → GPU 3, etc.). Multiple subagents may share the same GPU — MuJoCo rendering is lightweight.

**Do NOT check GPU occupancy with nvidia-smi.** Another experiment may be running in parallel on the same GPUs, and checking occupancy would cause you to skip usable GPUs.

### 3. Dispatch subagents

⚠️ **CRITICAL: Do NOT rewrite, paraphrase, or restructure the subagent prompt template.**

1. Read `.claude/robosuite/fix-loop/subagent-prompt.md`
2. Copy the template block (everything inside the ``` fences) **verbatim**
3. Fill in ONLY the four variables at the top: TASK, CONFIG, GPU, CAMERA_KEY
4. Pass the result as the `prompt` parameter — **no other modifications**

Do not add extra context, reorder sections, summarize sections, or omit any part of the template. The template contains precise formatting requirements (e.g. reasoning.txt format) that will be silently lost if you rewrite it. Any additional context (e.g. known failed seeds, existing fix code) must go in an `## Additional Context` section **appended after the template's last line**, never replacing or interleaved with template content.

```python
Agent(
    description="Robosuite fix loop: <task>",
    subagent_type="general-purpose",
    prompt=<filled-in template from .claude/robosuite/fix-loop/subagent-prompt.md>,
    run_in_background=True
)
```

Send all dispatches in one message. Then **stop** — you will be notified when each finishes.

### 4. Handling subagent notifications

When a `<task-notification>` arrives, check the `<result>` field to determine if it was a **crash** or a **successful completion**:

- **Crash/rate-limit:** `<result>` contains an error message (e.g. `API Error: Request rejected (429)`, or any non-findings text). → Go to **Step 4a**.
- **Successful completion:** `<result>` contains findings or a normal completion message. → Go to **Step 4b**.

#### Step 4a — On crash: auto-redispatch

1. Read the subagent prompt template from `.claude/robosuite/fix-loop/subagent-prompt.md`
2. Fill in TASK, CONFIG, GPU, CAMERA_KEY, and leave the rest of the subagent prompt template verbatim.
3. Dispatch a new background subagent — it will pick up prior progress via the checkpoint file at `/tmp/fix_progress_checkpoint_${TASK}.md`

**Do NOT** read trace.json, code.py, summary.txt, or the subagent's JSONL transcript. Only check disk state.

#### Step 4b — On completion: redispatch + update skills

When a subagent completes successfully:

1. `cat docs/progress/fix_loop_robosuite_progress.md` — verify progress updated
2. Update skills from `findings.md` for the completed task (see step 5)
3. Go idle again

### 5. Update skills

After each subagent completion, **you MUST read findings from the baseline task dir** (e.g. `outputs/baseline_robosuite_multimodel_ensemble_traced/ensemble_multimodel/ensemble_multimodel/<config_stem>/findings.md`) and promote **generalizable patterns** to the skill library.

| Skill | What to add |
|---|---|
| `.claude/robosuite/fix-loop/skills/localize.md` | New SAM3/Molmo prompts and strategies that worked for Robosuite objects, API failure fallbacks |
| `.claude/robosuite/fix-loop/skills/grasp.md` | Grasp strategies, gripper width thresholds, z_offset findings for Robosuite |
| `.claude/robosuite/fix-loop/skills/transport.md` | Waypoint sequences, bimanual coordination patterns, collision avoidance |

---

## Task Reference

| Task | Config | Camera | Notes |
|---|---|---|---|
| `cube_lifting` | `env_configs/robosuite/cube_lifting_multimodel_aspire_traced.yaml` | `robot0_robotview` | Single arm |
| `cube_restack` | `env_configs/robosuite/cube_restack_multimodel_aspire_traced.yaml` | `robot0_robotview` | Single arm |
| `cube_stack` | `env_configs/robosuite/cube_stack_multimodel_aspire_traced.yaml` | `robot0_robotview` | Single arm |
| `nut_assembly` | `env_configs/robosuite/nut_assembly_multimodel_aspire_traced.yaml` | `robot0_robotview` | Single arm |
| `spill_wipe` | `env_configs/robosuite/spill_wipe_multimodel_aspire_traced.yaml` | `robot0_robotview` | Single arm; sponge attachment |
| `two_arm_lift` | `env_configs/robosuite/two_arm_lift_multimodel_aspire_traced.yaml` | `robot0_robotview` | Bimanual |
| `two_arm_handover` | `env_configs/robosuite/two_arm_handover_multimodel_aspire_traced.yaml` | `robot0_robotview` | Bimanual handover |
