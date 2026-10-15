---
name: evosearch-subagent-prompt
description: Self-contained prompt template for Evolutionary Search multi-task actor subagents. Runs Evolutionary Search iterations on seeds 51–65, then runs Stage 2 (seeds 1–50) before returning. Coordinator just redispatches the freed GPU.
---

# Evolutionary Search Multi-Task Actor Subagent Prompt Template

Copy the block below, fill in the variables, and pass it as `prompt` to a background subagent.

> **Model:** Use a high-capability model. Evolutionary Search's multi-candidate reasoning needs
> trace diagnosis across 8 candidates and cross-iteration hypothesis refinement.

---

```
## Task Assignment

BENCHMARK:  <libero-pro|robosuite>
SUITE:      <full suite name>
TASK_ID:    <integer task id>
GPU:        <assigned simulation GPU>
TASKSHORT:  <short_unique_name_for_logs>
BASELINE_RATE: <e.g. "16% (50 seeds)">   # from the fix loop — what we're trying to beat
EVOSEARCH_DIR: outputs/evosearch             # change to outputs/evosearch_rerun for a clean rerun
EVALDIR:    outputs/evosearch_eval           # change to outputs/evosearch_eval_rerun for a clean rerun

Working directory: $REPO_ROOT (the CaP harness repository root)
Task workspace: TASK_DIR=$EVOSEARCH_DIR/$BENCHMARK/$SUITE/task_$TASK_ID

---

## ⛔ EVAL SET LOCKOUT — DURING ITERATIONS ONLY

**Seeds 1–50 are the held-out evaluation set. DO NOT run them during Evolutionary Search
iterations.**

- All Evolutionary Search iteration evaluations use seeds **51–65 only** as the debug set.
- Never run the eval seeds 1–50 during iterations, it contaminates the results.
- **Stage 2 (seeds 1–50): YOU run this after iterations converge and the final code is chosen,
  before returning.**
- **Every run uses `--init-mode seeded`** (the scripts default to it). Omitting it collapses
  seeds onto the 50 saved initial states and corrupts the seed protocol.

Violation invalidates the benchmark.

---

## What You Are

You are an Evolutionary Search debugging subagent. Your job:
1. Run Evolutionary Search-style iterative debugging on seeds 51–65 until convergence or plateau
2. Save the best code as `$TASK_DIR/evosearch_best_code.py`
3. **Run Stage 2: eval the best code on seeds 1–50**
4. Return a structured findings report with both Stage 1 and Stage 2 results

You have full tool access (Bash, Read, Write, Edit, Glob, Grep).
**`.venv-libero/bin/cap-harness`** — use the pinned benchmark environment for every episode.
You own GPU $GPU exclusively; run every episode with
`CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU`. Both are required: EGL device enumeration
ignores `CUDA_VISIBLE_DEVICES`, so omitting `MUJOCO_EGL_DEVICE_ID` renders on physical GPU 0 and
corrupts depth when that GPU is busy.

---

## Context

**Baseline:** The fix loop achieved $BASELINE_RATE on this task. Your target: exceed that
significantly, ideally ≥80% on seeds 51–65.

**Skill library:** shared robot skills live in `.claude/skills/iterative-debugging/skills/`.
Read before writing any candidates.

**Program rules:** generated programs cannot `import` anything and have no NumPy, filesystem,
or network access; set a top-level `result` variable; only the documented public tools and the
six injected constructors are available. Full contract: `docs/api-reference.md`.

**Services:** 200 = UP on 8114 (SAM3), 8115 (Contact-GraspNet), 8116 (PyRoki). Check before
running:
```bash
for p in 8114 8115 8116; do echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/openapi.json)"; done
```

---

## ⛔ FORBIDDEN APIs

Native simulator access or unwrapping (`env.handle.env`, `sim.data.*`, `sim.model.*`,
`sim.forward()`, `inner.parsed_problem`, `inner._eval_predicate`, `inner.obj_body_id`,
`env._step_once()`), raw reward/success inspection, manual physics stepping, reading
`.bddl`/`.xml`/`.urdf` asset files for geometry, and hardcoded world poses / object dimensions /
pixel locations / seed-specific behavior that perception can derive.

This contract applies unchanged when `BENCHMARK` is `libero-pro` or `robosuite` (and remains the
default for every simulation benchmark). LIBERO-specific examples include the listed `inner.*`
state and BDDL-derived geometry. Robosuite-specific examples also include `env.sim.*`, raw
model/data/robot/observable objects, `_check_success()`, and reward access, whether reached
directly or through a wrapper, alias, or reflection.

This is a provenance rule, not only an API-name ban:

- Every task-dependent object choice, grasp or target pose, object dimension, and pixel location
  must be derived at runtime through the allowed public perception and CaP APIs.
- Debugging may tune generic relative offsets, tolerances, quantiles, and step limits. It must
  never copy observed coordinates, dimensions, pixel locations, raw reward/success values, or
  seed identities into executable constants or branches.
- Before saving `evosearch_best_code.py`, audit every task-dependent constant. If a candidate or
  its strategy used forbidden information, quarantine that candidate and all results derived
  from it; never rewrite it to hide its provenance or run it in Stage 2.

---

## Stage 1: Evolutionary Search Iterations on Seeds 51–65

### Step 0 — Check if evosearch_best_code.py already exists

```bash
ls $TASK_DIR/evosearch_best_code.py 2>/dev/null && echo EXISTS || echo MISSING
```

If EXISTS → audit the existing program under the non-privileged provenance rules above. Skip
to **What to Return** only if it passes; otherwise quarantine it and mark the task BLOCKED.

---

### Step 0b — Task language check (MANDATORY)

**The task name might not match the actual goal.** Always read the `task_language` reported by
the scene snapshot (Step 2). If it differs from the task name, **task language is ground
truth** — base all strategy on it. Record the actual language in `task_analysis.md` before
writing any candidates. (LIBERO `_task` suites remap the goal; the BDDL-derived name is
misleading.)

---

### Step 1 — Read skills and existing baseline

```bash
cat .claude/skills/iterative-debugging/skills/grasp.md
cat .claude/skills/iterative-debugging/skills/localize.md
cat .claude/skills/iterative-debugging/skills/transport.md
cat .claude/skills/iterative-debugging/skills/manipulation.md
```

**Also read the existing fix_code.py if it exists** — this is your baseline to beat and **must
be seeded as candidate_A**:
```bash
cat $TASK_DIR/evosearch_best_code.py 2>/dev/null || \
cat outputs/runs/LATEST/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py 2>/dev/null || \
cat "$(ls -t outputs/runs/*/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py 2>/dev/null | head -1)" 2>/dev/null || \
cat outputs/iterative_debugging/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py 2>/dev/null || \
echo "No baseline fix_code.py found"
```

**If a baseline code is found:** Record its path in `task_analysis.md`. Seed it as candidate_A
verbatim — do not modify it.

Also read [skills/evosearch-iteration.md](skills/evosearch-iteration.md) — §8 carries the
motion-call budget, contact-task patterns, and arm-blocking recovery guidance.

---

### Step 2 — Create run directory and scene snapshot

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$TASK_DIR/$RUN_ID
mkdir -p $RUN_DIR
```

**Scene snapshot** (REQUIRED before writing any candidates):
```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
.venv-libero/bin/cap-harness run \
  --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --seed 51 \
  --program .claude/skills/iterative-debugging/scripts/scene_snapshot.py \
  --output-root $RUN_DIR/snapshot --init-mode seeded
```

The command prints `run: <run_dir>`. Read from that run with the Read tool:
- `source/program-result.json` — authoritative `task_language`, cameras, task metadata
- `media/keyframes/` — lossless RGB of the untouched scene from every camera (wide agentview
  for layout, eye-in-hand for close-up)

To validate SAM3 prompts, copy `scene_snapshot.py` into `$RUN_DIR`, fill `PROBE_PROMPTS`, re-run,
and read the probe scores plus `media/overlays/`. Fix any weak/failing prompts before writing
candidates.

Then populate `$RUN_DIR/task_analysis.md` (§1–5 from the images):
```bash
cat > $RUN_DIR/task_analysis.md << 'EOF'
# Task Analysis: <task_name>

## 1. Object Shape
...

## 2. Grasp/Approach Strategy
...

## 3. Goal Geometry
...

## 4. Placement/Movement Strategy
...

## 5. Hypotheses for iter_00
- Baseline code achieves $BASELINE_RATE — what is it doing wrong?
- Blocked approach directions visible in snapshot:
- Uncertain geometry assumptions:

## 6. Iteration Log
EOF
```

---

### Step 3 — Write K=8 candidates for iter_00

**Always seed candidate_A from the existing fix_code.py baseline** (if it exists) — this gives
a concrete performance floor and shows where the baseline fails.

Each candidate must test a distinct hypothesis. No two candidates should fail at the same stage
for the same reason.

```bash
mkdir -p $RUN_DIR/iter_00/candidate_{A,B,C,D,E,F,G,H}
# Write candidate_A/code.py through candidate_H/code.py
# Each has a docstring: Hypothesis / Differs from prior / Expected failure if wrong
```

See [skills/evosearch-iteration.md](skills/evosearch-iteration.md) §8 for the K=8 diversity
rule and template.

---

### Step 4 — Iteration eval (seeds 51–65)

```bash
python3 .claude/skills/evosearch/scripts/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_00 \
    --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID \
    --trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 \
    --sim-gpus $GPU --parallel-per-gpu 2 \
    --cap-harness .venv-libero/bin/cap-harness \
    2>&1 | tee $RUN_DIR/iter_00/eval.log
```

**Set `run_in_background=True`** on this Bash call. Results (15 trials per candidate) are saved
under `$RUN_DIR/iter_00/candidate_X/eval/` and aggregated in `iter_summary.json`.

Read leaderboard after completion:
```bash
python3 .claude/skills/evosearch/scripts/analyze_evosearch_traces.py \
    --iter-dir $RUN_DIR/iter_00 --summary-only
```

Analyze traces:
```bash
python3 .claude/skills/evosearch/scripts/analyze_evosearch_traces.py --iter-dir $RUN_DIR/iter_00
```

---

### Step 6 — Iterate

**Stopping criteria — stop Stage 1 when ANY of these is true:**
- Best candidate ≥ **80%** on seeds 51–65 → **SOLVED**
- **5 iterations** completed

**When progress stalls:** Do NOT stop early just because recent iterations showed small gains.
Use the remaining iteration budget to explore structurally new approaches — different grasp
strategies, different localization methods, different motion primitives. A plateau on the
current approach family means the current family is wrong, not that the task is unsolvable.

**Generalization rule — think eval seeds, not debug seeds:**

The 15 debug seeds are a small, potentially unrepresentative sample. The code will be evaluated
on 50 unseen seeds — write for those, not for the 15. Prefer strategies that work for
mechanistic reasons (correct object identity, robust localization, physically sound grasp
geometry) over strategies that score well by exploiting patterns specific to seeds 51–65. Avoid
hard-coded thresholds, image-region masks, or XY offsets that were derived by fitting to
observed debug-seed failures — these rarely transfer.

**Per iteration:**
1. Update `task_analysis.md` §1–5 with any new geometry from keyframes
2. Append `### iter_NN` to §6 (leaderboard table + eliminated hypotheses + open questions)
3. Write K=8 new candidates seeded from top-3 survivors of the current iteration
4. Eval all 8 candidates on seeds 51–65 (same command as Step 4, adjusted iter dir)
5. Read `analyze_evosearch_traces.py` output — watch for failed motion spans, empty
   segmentation prompts, and wall-time anomalies (blocking)

```bash
# Iter N+1 setup
mkdir -p $RUN_DIR/iter_NN
# Write candidates, then eval as in Step 4
```

Use the same `--trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65` across all iterations
(cross-iteration comparison is fair since seeds are fixed).

---

### Step 7 — Save best code, run Stage 2, write findings

Before Step 7a, perform the non-privileged provenance audit above on the selected candidate. Only
a candidate that passes this audit may become `evosearch_best_code.py` or proceed to Stage 2. If
it fails, preserve the evidence, mark the task BLOCKED, and report the result as quarantined.

**7a — Save best code:**
```bash
BEST_CANDIDATE="candidate_X"   # from final iter leaderboard
BEST_CODE="$RUN_DIR/iter_NN/candidate_X/code.py"

# Sanity check: Evolutionary Search best must beat fix_code.py (candidate_A = fix_code verbatim)
# on seeds 51–65. If not, fall back to fix_code.py so Stage 2 uses the stronger baseline.
python3 - "$RUN_DIR" << 'PYEOF'
import json, sys
from pathlib import Path

rdir = Path(sys.argv[1])
baseline, best_rate = 0.0, 0.0
for f in sorted(rdir.rglob("iter_summary.json")):
    for c in json.loads(f.read_text()).get("candidates", []):
        best_rate = max(best_rate, c["pass_rate"])
        if c["candidate"] == "candidate_A":
            baseline = max(baseline, c["pass_rate"])

print(f"Best Evolutionary Search: {best_rate:.0%}  (seeds 51-65)")
print(f"Baseline fix_code.py:     {baseline:.0%}  (candidate_A verbatim)")
sys.exit(0 if best_rate > baseline else 1)
PYEOF

if [ $? -ne 0 ]; then
    echo "⚠ Evolutionary Search did not beat fix_code.py — using fix_code.py as evosearch_best_code.py"
    BEST_CODE="outputs/runs/LATEST/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py"
    if [ ! -f "$BEST_CODE" ]; then
        # LATEST may point at a newer campaign that never debugged this task;
        # fall back to the newest fix_code.py across all campaign roots.
        BEST_CODE="$(ls -t outputs/runs/*/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py 2>/dev/null | head -1)"
    fi
    if [ ! -f "$BEST_CODE" ]; then
        BEST_CODE="outputs/iterative_debugging/$BENCHMARK/$SUITE/task_$TASK_ID/fix_code.py"
    fi
    if [ ! -f "$BEST_CODE" ]; then
        echo "ERROR: fix_code.py not found at $BEST_CODE — marking BLOCKED"
        touch $TASK_DIR/BLOCKED
        exit 0
    fi
fi

cp "$BEST_CODE" $TASK_DIR/evosearch_best_code.py
mkdir -p outputs/working_codes
cp "$BEST_CODE" "outputs/working_codes/${BENCHMARK}_${SUITE}_task_${TASK_ID}_evosearch.py"
```

If all approaches failed → mark BLOCKED and stop:
```bash
touch $TASK_DIR/BLOCKED
```
(skip Stage 2 if BLOCKED)

---

**7b — Stage 2: eval best code on seeds 1–50**

Use the shared immutable-manifest validation script — it resumes safely and records one
manifest per (code, settings, seeds) identity:

```bash
nohup python3 .claude/skills/iterative-debugging/scripts/run_validation.py \
    --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --gpu $GPU \
    --program $TASK_DIR/evosearch_best_code.py \
    --output-root $EVALDIR \
    --cap-harness .venv-libero/bin/cap-harness \
    --seeds $(seq 1 50) --resume \
    > /tmp/evosearch_s2_${TASKSHORT}.log 2>&1 &
echo $! > /tmp/evosearch_s2_${TASKSHORT}.pid
```

Poll every ~5 min with a short Bash call until the final `run=` line appears:
```bash
tail -3 /tmp/evosearch_s2_${TASKSHORT}.log
grep -c "^seed" /tmp/evosearch_s2_${TASKSHORT}.log
```

Repeat this poll call until the log ends with `run=<id> status=complete passes=<N>/50`. Do not
run other Bash commands between polls — just wait and re-poll. If the process dies early,
re-run the same command with `--resume` (it skips completed seeds).

Read final results from the manifest printed on the last line
(`manifest=<path>`): `passes`, `trials`, `pass_rate`.

---

**7c — Write findings.md:**
```bash
cat > $TASK_DIR/findings.md << 'EOF'
## Task: $BENCHMARK / $SUITE / task_$TASK_ID
## Baseline rate (fix loop): $BASELINE_RATE
## Best Evolutionary Search rate (seeds 51–65): <N/15> (<pct>%)
## Stage 2 (seeds 1–50): <N/50> (<pct>%)

### Evolutionary Search Run
- Run dir: $RUN_DIR
- Iterations completed: <N>
- Stopping reason: <solved|max_iterations>

### What Fixed It vs Baseline
- <key strategy change>

### SAM3 Prompts That Worked
| Object | Prompts | Notes |
|---|---|---|

### Failure Modes Eliminated
- <approach that failed + why>

### Generalizable Patterns
- <anything worth adding to skill library>

### Non-privileged API audit
- <PASS, or QUARANTINED with the forbidden dependency and all affected artifacts/results>

### Skill Library Updates Made
- <if any skills were updated during this run>
EOF
```

---

## What to Return

```
BENCHMARK: <benchmark>
SUITE: <suite>
TASK_ID: <task id>
GPU: <N>
Baseline rate: <from fix loop>

Stage 1 (seeds 51–65):
  Best candidate: <candidate_X> at iter_NN
  Best pass rate: <N>/15 (<pct>%)
  Stopping reason: <solved|max_iterations|BLOCKED>
  Iterations run: <N>
  Run dir: $TASK_DIR/<run_id>/

Stage 2 (seeds 1–50): <N>/50 (<pct>%)
Non-privileged API audit: PASS/QUARANTINED (<reason if quarantined>)

Key findings (3 bullets):
  - <strategy that worked vs baseline failure mode>
  - <SAM3 prompts / grasp parameters>
  - <any generalizable pattern for skill library>
```
```
