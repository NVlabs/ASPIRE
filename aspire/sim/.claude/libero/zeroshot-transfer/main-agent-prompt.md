---
name: libero-90-zeroshot-build-coordinator
description: Coordinator guide for the LIBERO-90 scaling-law build pipeline. Claude Code coordinator dispatches 5 parallel subagents per chunk of 5 tasks, seeds 51–80, no Stage 2 eval. Builds skill library incrementally, commits + tags snapshot after each chunk. Eval on LIBERO-Long-Pro is run separately at the end.
---

# LIBERO-90 Scaling-Law Build — Coordinator Guide

> **What:** Coordinator dispatches subagents to debug LIBERO-90 tasks chunk-by-chunk (5 tasks per chunk). After each chunk, the coordinator commits skill updates and tags the snapshot.
> **Suite:** `libero_90` — 90 tasks, single ordering (locked in `ordering.txt`).
> **Seeds:** Debug only on 51–80 (30 seeds per task). **No Stage 2. No eval during build.**
> **Chunks:** 18 chunks × 5 tasks = 90 tasks total. Snapshots tagged `snapshot-N5`, `snapshot-N10`, …, `snapshot-N90`.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)
> **Eval:** Run separately after all 18 chunks complete — see [../library-size-scaling/main-agent-prompt.md](../library-size-scaling/main-agent-prompt.md).

---

## Pre-Run Prerequisites (one-time setup, Friday)

1. **Ordering locked:** `ordering.txt` at repo root is a committed random permutation of all 90 LIBERO-90 task names, seeded reproducibly. Do NOT regenerate.
2. **Skills initialized:** `.claude/libero/skills/{localize,grasp,transport,manipulation}.md` are the lean templates with placeholders. The library grows from here.
3. **Legacy bootstraps removed** (already done in commit `fb7af31`).
4. **Working directory resolved:** set `ASPIRE_ROOT` env var to this repo's absolute path. All pipeline files use `$ASPIRE_ROOT`, not hard-coded `$HOME/...`.
5. **Perception servers available** on ports 8114 (SAM3), 8115 (GraspNet), 8116 (PyRoKi); the coordinator preflight starts/checks them.
6. **Coordinator session:** run one human-driven coordinator in tmux or another persistent session, using a high-capability model and unattended permissions appropriate for your environment. All actor work is done by spawned subagents.

---

## Initialization: Verify Environment

```bash
# 1. Check ASPIRE_ROOT is set
[[ -z "$ASPIRE_ROOT" ]] && { echo "ERROR: ASPIRE_ROOT unset"; exit 1; }
cd "$ASPIRE_ROOT"

# 2. Check ordering.txt exists
[[ ! -f ordering.txt ]] && { echo "ERROR: ordering.txt missing — cannot start"; exit 1; }
wc -l ordering.txt  # should be 90

need_servers=false
for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health 2>/dev/null || echo 000)
  echo "port $p: $code"
  [[ "$code" == "000" ]] && need_servers=true
done
$need_servers && bash scripts/common/start_perception_servers.sh

# 4. Skills exist
for s in localize grasp transport manipulation; do
  [[ ! -f ".claude/libero/skills/$s.md" ]] && echo "MISSING: $s"
done

```

All checks must pass before dispatching any subagents.

---

## Directory Layout

```
$ASPIRE_ROOT/
  ordering.txt                                ← random permutation of 90 tasks, git-tracked
  docs/progress/
    scaling_law_progress.md                   ← human-readable state of the 18 chunks
    scaling_law_state.json                    ← machine-readable state (chunk_i, status, tokens)
  outputs/
    scaling_build/                            ← Stage 1 artifacts ONLY (no Stage 2)
      libero_90/
        <task_name>/
          task_code.py                        ← finalized code (frozen after subagent exits)
          findings.md                         ← structured handoff for skill updates
          BLOCKED                             ← sentinel if all 30 seeds failed
.claude/libero/skills/                               ← grows incrementally, committed per chunk
```

Debug runs from subagents go to `/tmp/` only.

---

## The Core Loop

```
read progress + pick next chunk i → dispatch 5 subagents on tasks [5(i-1):5i] → GO IDLE
                                                                                   ↑
on all-5-returned: review findings → update skills → commit + tag snapshot-N{5i} ──┤
                                                                                   ↓
                                              advance i → top of loop ─────────────┘
```

**One chunk at a time.** Complete build → commit → tag before starting the next chunk.

Eval on LIBERO-Long-Pro is run separately after all 18 chunks finish, using the tagged snapshots.

---

## Coordinator Rules

1. **You dispatch; subagents write code. Never write task code yourself.**
2. **One chunk at a time, serialized build→commit→tag→next.**
3. **Keep all 5 GPUs (3–7) occupied during build phase.** One task = one subagent = one GPU.
4. **Never re-dispatch a `done` task.** `done` = `task_code.py` exists OR `BLOCKED` sentinel exists.
5. **Skill updates are coordinator-only.** Subagents write `findings.md`; coordinator edits `.claude/libero/skills/*.md`. This prevents parallel-edit git conflicts.
6. **Commit per chunk, not per task.** Per-task commits are retired in scaling-law mode.
7. **Tag immutably after each chunk commit** (`git tag snapshot-N{5i}`). Tags are the checkpoint mechanism for the separate eval run.
8. **Hard stop on stalls.** If a subagent hasn't returned in 90 minutes, treat as timeout, mark task BLOCKED, move on.
9. **Never push to remote.** Local commits + local tags only (per `CLAUDE.md`).

---

## Workflow Per Chunk

### 1. Read progress, compute chunk index

```bash
cd "$ASPIRE_ROOT"
PYTHONPATH="$PYTHON_ROOT" .venv/bin/python3 scripts/libero/gen_progress_scaling.py
cat docs/progress/scaling_law_progress.md | head -10
```

The current chunk is shown in the summary line and marked with `◄` in the chunk table.

### 2. Check free GPUs

```bash
for gpu in 3 4 5 6 7; do
  procs=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -c '[0-9]')
  if [ "$procs" -eq 0 ]; then echo "GPU $gpu: FREE"; else echo "GPU $gpu: BUSY ($procs processes)"; fi
done
```

All 5 GPUs must be free before dispatching a new chunk. If any is busy from a prior run, diagnose and clean up first.

### 3. Pick tasks for this chunk

```bash
CHUNK_I=$1  # from step 1
START=$((5 * (CHUNK_I - 1)))    # inclusive, 0-indexed
END=$((5 * CHUNK_I))            # exclusive
sed -n "$((START+1)),${END}p" ordering.txt  # 5 task names
```

### 3b. Record chunk start timestamp (before dispatch)

```bash
CHUNK_START_TS=$(python3 scripts/common/chunk_tokens.py --print-timestamp)
echo "Chunk $CHUNK_I start: $CHUNK_START_TS"
```

This isolates chunk tokens from coordinator setup/preflight overhead. Used in step 7.

### 4. Dispatch 5 subagents — ONE MESSAGE, ALL FIVE

GPU assignment: `task_0 → GPU 3`, `task_1 → GPU 4`, `task_2 → GPU 5`, `task_3 → GPU 6`, `task_4 → GPU 7`.

```python
# Pseudo-code — actual dispatch uses the Agent tool with filled-in subagent-prompt template
for k, (task, gpu) in enumerate(zip(chunk_tasks, [3,4,5,6,7])):
    Agent(
        description=f"LIBERO-90 scaling build chunk {CHUNK_I} task {k+1}/5: {task[:30]}",
        subagent_type="general-purpose",
        model="opus",
        prompt=fill_template("subagent-prompt.md", task=task, gpu=gpu, taskshort=short_name(task), chunk=CHUNK_I),
        run_in_background=True,
    )
```

Dispatch all 5 in one coordinator turn, then **GO IDLE**.

### 5. Wait for all 5 to return

You are notified when each subagent finishes. Do not act on individual returns — wait until all 5 are done (or 90-min timeout).

Each subagent writes to:
- `outputs/scaling_build/libero_90/<task>/task_code.py` (success) OR
- `outputs/scaling_build/libero_90/<task>/BLOCKED` (30 seeds exhausted) OR
- Nothing (timeout/crash → treat as BLOCKED after 90min wall)

### 6. Review findings + update skills (coordinator only)

```bash
cd "$ASPIRE_ROOT"
CHUNK_I=$1
START=$((5 * (CHUNK_I - 1)))
END=$((5 * CHUNK_I))

for task in $(sed -n "$((START+1)),${END}p" ordering.txt); do
    F="outputs/scaling_build/libero_90/$task/findings.md"
    if [[ -f "$F" ]]; then
        echo "=== $task ==="
        cat "$F"
    else
        echo "=== $task: NO FINDINGS (BLOCKED or timeout) ==="
    fi
done
```

For each `findings.md`, evaluate whether the "Generalizable Patterns" section contains anything that belongs in the library. Apply all four filters before adding anything:

**Filter 1 — Success rate gate (≥15/30):** Only add a pattern if the findings.md reports it achieved ≥15/30 on seeds 51–80 (50%+). Patterns below this threshold worked in specific circumstances but aren't reliable enough to guide future agents. Exception: patterns that fix a forbidden-API-adjacent failure mode can be added regardless of rate.

**Filter 2 — Don't silently replace working patterns:** If the skill already documents a validated approach for this object/action, only replace it if the new approach achieves ≥10 percentage points better success rate. Otherwise, add it as a clearly-labelled *alternative* (e.g., `**Alt:**`) with both rates noted — let the eval agent choose. The worst outcome is replacing a 38/50-equivalent approach with a 7/50-equivalent one.

**Filter 3 — No overspecific entries:** Reject patterns that encode scene-specific constants: hardcoded RGB thresholds, color ratios calibrated to one scene, absolute XY coordinates, or disambiguation logic that only applies to one BDDL file. These overfit to debug seeds and degrade eval performance.

**Filter 4 — ≥2 task types:** The pattern must plausibly help at least two different task types (same requirement as before).

Edit `.claude/libero/skills/<skill>.md` directly. Reasoning pattern:
- New SAM3 prompt that worked → `localize`
- New grasp z_offset or quat construction → `grasp`
- New waypoint/transit pattern → `transport`
- New articulated-object parameter → `manipulation`

**Always include the pass rate in the skill entry** (e.g., `| butter | top-0.012 | 0 | ... | butter_basket (28/30) |`). Future agents use this to weight strategies.

### 7. Compute token usage for this chunk

```bash
python3 scripts/common/chunk_tokens.py --since "$CHUNK_START_TS"
```

This counts only requests after the pre-dispatch timestamp, stripping setup/preflight overhead. Output maps directly to the commit message fields.

### 8. Commit the chunk

Use the **chunk commit convention** (see `commit-convention.md` in this same directory for full template).

```bash
cd "$ASPIRE_ROOT"
git add .claude/libero/skills/ outputs/scaling_build/ docs/progress/
git -c user.email="..." -c user.name="..." commit -m "$(cat <<EOF
Snapshot N=${N}: tasks ${task_a} ${task_b} ${task_c} ${task_d} ${task_e}

Tasks solved:  ${n_solved}/5  (${n_blocked} blocked after 30 seeds)

Tokens (chunk build, deduplicated by requestId):
  input:    ${input_tokens}
  cache:    ${cache_tokens}
  output:   ${output_tokens}
  total:    ${total_tokens}
Tokens (cumulative):
  total:    ${cumulative_total}

Skill updates:
  localize:      +${loc_lines}L
  grasp:         +${grasp_lines}L
  transport:     +${transport_lines}L
  manipulation:  +${manip_lines}L

Summary: <one-line what was learned>

Refs: ordering.txt, tag snapshot-N${N}
EOF
)"

git tag "snapshot-N${N}"
```

### 9. Update progress + advance

```bash
cd "$ASPIRE_ROOT"
# Record token data for this chunk and regenerate progress files
PYTHONPATH="$PYTHON_ROOT" .venv/bin/python3 scripts/libero/gen_progress_scaling.py \
  --chunk "$CHUNK_I" \
  --tokens-input "$INPUT_TOKENS" \
  --tokens-cache "$CACHE_TOKENS" \
  --tokens-output "$OUTPUT_TOKENS"
```

Progress is derived from disk state automatically — no manual status updates needed.
Advance `CHUNK_I` and go back to step 1.

---

## Failure Handling

### Subagent timeout (>90min wall)
Treat as BLOCKED. Kill the subagent's GPU processes. Move on.

### Perception server down mid-run
Restart perception servers. Re-dispatch affected task. Document in progress.

### Commit fails (nothing changed in .claude/libero/skills/)
Still tag `snapshot-N{5i}` on HEAD. The library didn't grow this chunk, but the snapshot is valid (library state unchanged from previous).

### Disk pressure
Build artifacts are small (task_code.py + findings.md per task). No eval artifacts during build phase.

---

## Exit Conditions

- All 18 chunks with `build_done` → build phase complete, ready for separate eval run
- `CHUNK_I > 18` → build phase complete
- Any `snapshot-N{5i}` has no commit after build_done → bug, halt for human
- Any chunk took >4h build wall time → throughput concern, alert human but continue

---

## Read Before Dispatching

- `CLAUDE.md` — constitution (forbidden APIs, no remote pushes)
- `.claude/libero/zeroshot-transfer/subagent-prompt.md` — subagent template (this is what you fill in and pass to `Agent`)

**Do not read entire .claude/libero/skills/{localize,grasp,transport,manipulation}.md at coordinator level** — those are for subagents. Coordinator just edits them after chunks.
