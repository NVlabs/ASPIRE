# Fix Loop — Coordinator Agent Guide

> **What:** Each task goes through two stages. Stage 1 (a subagent): explore the scene once, generate initial code without an external baseline, debug seeds 51–65, select one fix. Stage 2 (a script you run): validate the selected fix on held-out seeds 1–50.
> **Progress:** `docs/progress/fix_loop_progress.md` — single source of truth for task status; always regenerate it with `scripts/libero/gen_progress.py` before reading it.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)
> **Reruns:** before rerunning any task or suite, follow [clean-task-slate.md](clean-task-slate.md).

---

## Task Lifecycle

Every task moves through exactly these states (as reported by `gen_progress.py`):

```
pending ──(dispatch Stage 1 subagent)──► stage1-done ──(run Stage 2 eval script)──► done
```

- **`pending`** → dispatch a subagent (Stage 1). Never run Stage 2 on a pending task.
- **`stage1-done`** → `fix_code.py` exists; run the Stage 2 validation **script** yourself, in the background. **Never dispatch a subagent for a `stage1-done` task** — it would redo Stage 1 from scratch and could overwrite a good `fix_code.py`.
- **`done`** → all 50 held-out seeds on disk. **NEVER touch a `done` task.** Re-dispatching or re-evaluating creates duplicate trial dirs and corrupts benchmark results.

---

## GPU Ownership Ledger

You are the only allocator of GPUs 3–7. Keep an explicit ledger in your working notes, e.g.:

```
GPU 3: SUBAGENT  libero_goal_swap/open_the_middle_drawer      (dispatched)
GPU 4: EVAL      libero_object_task/pick_up_the_milk           (background, started)
GPU 5: FREE
...
```

Rules:

1. **One job per GPU, ever.** A job is either a Stage 1 subagent or a Stage 2 eval.
2. **A GPU stays owned by its task from subagent dispatch until that task's Stage 2 eval finishes.** When a subagent returns, immediately start Stage 2 for that task on the **same GPU** — do not give that GPU to a new subagent until the eval completes.
3. **Never trust `nvidia-smi` to decide a GPU is free.** Subagents idle between replays and the eval script spawns a fresh process per seed, so an owned GPU often shows 0 processes. `nvidia-smi` is a sanity check only:

   ```bash
   for gpu in 3 4 5 6 7; do
     procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
     if [ "$procs" -eq 0 ]; then echo "GPU $gpu: no processes"; else echo "GPU $gpu: $procs processes"; fi
   done
   ```

   Use it (a) at session start, to confirm nothing stale is running before you claim GPUs, and (b) as a fallback when a subagent's result is missing its `GPU:` line — cross-check against your ledger.
4. If you are resuming a session and have no ledger, rebuild it: regenerate progress, check `nvidia-smi`, and check for running background tasks before dispatching anything.

---

## Initialization: Verify Perception Servers

Before dispatching any subagents, confirm all three servers are up (404 = UP, 000 = DOWN):

```bash
for p in 8114 8115 8116; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

| Port | Server | GPU |
|---|---|---|
| 8114 | SAM3 | 0 |
| 8115 | GraspNet | 1 |
| 8116 | PyRoKi | 2 |

If any server is down, start it before proceeding — subagents will fail silently without them.

---

## The Loop

```
regenerate+read progress → fill GPUs 3–7 (pending→subagent, stage1-done→eval) → GO IDLE
        ↑                                                                          │
        │   on SUBAGENT completion: start Stage 2 eval on that same GPU,           │
        │                           update skills + record promotion, go idle      │
        │   on EVAL completion:     GPU is now free → assign it the next task,     │
        │                           go idle                                        │
        └──────────────────────────────────────────────────────────────────────────┘
```

Repeat until every task is `done`.

---

## Coordinator Rules

1. **Dispatch subagents — never debug yourself.** Your only jobs: dispatch subagents, run Stage 2 evals, update skills, keep the ledger.
2. **Go idle after dispatching.** You are notified automatically when a subagent or background eval finishes. Do not poll, monitor processes, or watch GPUs while jobs are running.
3. **Keep all 5 GPUs (3–7) occupied** — with subagents *or* evals, one job per GPU (see ledger rules). The skill-promotion gate takes priority over immediately dispatching another Stage 1 task.
4. **Never read task-specific debug files.** Do not open `trace.json`, `code.py`, `summary.txt`, or keyframe images — that's subagent work. You may read `findings.md` (and skim `fix_code.py` if needed) only when writing a skill update.
5. **Route by state:** `pending` → subagent; `stage1-done` → Stage 2 eval script; `done` → never touch. Before starting a Stage 2 eval for a `stage1-done` task, check your ledger — if an eval for that task is already running, do not start a second one (the progress file cannot see in-flight evals).

---

## Workflow

### 1. Regenerate and read progress

Always regenerate before reading — the file on disk is stale otherwise:

```bash
PYTHONPATH="$PYTHON_ROOT" .venv-libero/bin/python3 scripts/libero/gen_progress.py
cat docs/progress/fix_loop_progress.md
```

### 2. Fill every free GPU

For each GPU your ledger shows FREE, assign one job:

- Next `pending` task → dispatch a Stage 1 subagent (step 3).
- Next `stage1-done` task with no eval in flight → start a Stage 2 eval in the background (see Stage 2 section).

Update the ledger, send all dispatches in one message, then **stop and go idle**.

### 3. Dispatching a Stage 1 subagent

Copy the template from [subagent-prompt.md](subagent-prompt.md), fill in SUITE, TASK, GPU:

```python
Agent(
    description="Fix loop: <suite>/<task>",
    subagent_type="general-purpose",
    prompt=<filled-in template from subagent-prompt.md>,
    run_in_background=True
)
```

Always pass the **full suite name** (see collision warning below).

### 4. On SUBAGENT completion

The result includes `GPU: <N>` (if missing, use your ledger + the nvidia-smi fallback). Then, in order:

1. Regenerate progress (step 1) and confirm the task now shows `stage1-done`.
2. **Start the Stage 2 eval for this task, on this same GPU, in the background** (see Stage 2 section). The GPU remains owned by this task — do NOT dispatch a new subagent on it.
3. Update skills from this task's `findings.md` and record the promotion (step 6).
4. Update the ledger (`SUBAGENT` → `EVAL`) and go idle.

### 5. On EVAL completion

1. Regenerate progress and confirm the task shows `done` (50/50 results). If seeds are missing, re-run the same eval command with `--resume` on the same GPU.
2. Verify this task's skill-promotion record (step 6). Do not assign another Stage 1 task to the GPU if verification fails.
3. The GPU is now genuinely free — mark it FREE in the ledger and assign it the next job (step 2).
4. Go idle.

### 6. Update skills

After each subagent completion, read
`outputs/libero_fix_loop/$SUITE/$TASK/findings.md` and promote **generalizable patterns** to the skill library. Subagents never write to skills — only the coordinator does. If `findings.md` is missing, note it and use the subagent's returned summary instead.

Snapshot the library before editing, then record the exact per-task patch afterwards:

```bash
.venv/bin/python3 scripts/libero/record_skill_promotion.py begin \
  --suite "$SUITE" --task "$TASK"

# Read findings.md and update .claude/libero/skills/*.md.

.venv/bin/python3 scripts/libero/record_skill_promotion.py finish \
  --suite "$SUITE" --task "$TASK"
```

If there is nothing generalizable to promote, leave the library unchanged and pass a concise
`--reason` to `finish`. Before dispatching another Stage 1 task on this task's GPU, require:

```bash
.venv/bin/python3 scripts/libero/record_skill_promotion.py verify \
  --suite "$SUITE" --task "$TASK"
```

The append-only ledger is
`outputs/libero_fix_loop/$SUITE/skill_promotions.jsonl`; exact patches and before snapshots are
stored under `outputs/libero_fix_loop/$SUITE/skill_promotions/`.

Route by topic: SAM3 prompts and disambiguation → [../skills/localize.md](../skills/localize.md);
grasp selection, offsets, verification → [../skills/grasp.md](../skills/grasp.md); waypoints, transit,
placement → [../skills/transport.md](../skills/transport.md); drawer/knob/push techniques →
[../skills/manipulation.md](../skills/manipulation.md).

**Write each pattern in the richest form that fits — do not default to a table row.**

- **Default format: a freeform subsection**, modeled on the existing ones ("Pre-Probe IK
  Conditioning" in transport.md, "Disambiguation" in localize.md):
  - **Trigger** — the symptom or scene condition that calls for the pattern.
  - **Code** — the working snippet (5–20 lines). Open the task's `fix_code.py` (and the snippet
    in `findings.md`) and extract the real code, generalizing task-specific prompts and
    constants into placeholders. Do not paraphrase code into prose.
  - **Why it works + evidence** — one or two sentences, plus provenance: source suite/task,
    which development seeds it fixed, the source `fix_code.py` path, and date.
- **Table rows are only for one-line lookups** — a prompt string, a Z formula, a numeric
  threshold. If the Notes cell needs a sentence of procedure, it is not a table row: write a
  subsection (optionally with a table row pointing to it).
- **Merge, don't append.** If an existing section already covers the pattern, extend it — add
  the new evidence, widen the trigger, note the variant. Never add a table row that paraphrases
  an existing section.

Only Stage 1 evidence may drive skill edits. Report held-out outcomes separately; never use them
to revise the shared library.

---

## Stage 2: Held-Out Evaluation (seeds 1–50) — run by the coordinator

Run this yourself, in the background, on the GPU the task already owns (`$GPU` below = that GPU). Subagents never run Stage 2.

```bash
.venv-libero/bin/python3 scripts/libero/run_fix_loop_validation.py \
  --suite "$SUITE" --task "$TASK" --gpu "$GPU" \
  --fix-code "outputs/libero_fix_loop/$SUITE/$TASK/fix_code.py" \
  --output-dir outputs/libero_fix_loop_eval \
  --seeds $(seq 1 50) --resume
```

Regenerate progress after it finishes. A task is `done` only with all 50 held-out results on disk.

---

## Suite Name Collision Warning

All six swap/task suite pairs share identical task names:
- `libero_goal_swap` ↔ `libero_goal_task`
- `libero_object_swap` ↔ `libero_object_task`
- `libero_spatial_swap` ↔ `libero_spatial_task`

Always include the full SUITE name when briefing a subagent.
- **`_swap` suites:** object positions randomized per seed — SAM3 handles naturally
- **`_task` suites:** language goal remapped — BDDL filename misleading, always use `env.handle.task_language`
