---
name: robosuite-evosearch-subagent-prompt
description: Per-task Robosuite Evolutionary Search prompt. Runs K=8 candidates on seeds 101-125, selects the best program, then evaluates it on seeds 1-100.
---

# Evolutionary Search — Robosuite Subagent Prompt Template

Copy the entire fenced block, fill only the assignment variables, and pass it
to one background subagent.

```
## Task Assignment

TASK:          <nut_assembly|two_arm_lift>
CONFIG:        <traced yaml from the coordinator task table>
FIX_CODE:      <tracked baseline from the coordinator task table>
GPU:           <simulation GPU>
BASELINE_RATE: <9/100|approximately 70/100>
EVOSEARCH_DIR: outputs/robosuite_evosearch
EVALDIR:       outputs/robosuite_evosearch_eval

Working directory: $ASPIRE_ROOT

## Filesystem and Security Boundary

Only access files under `$ASPIRE_ROOT` and task-owned scratch files under
`/tmp/robosuite_evosearch_${TASK}/`. Do not inspect home directories, secret
files, unrelated checkouts, or another task's scratch/output tree.

Generated Python is untrusted and the trial process is not a hardened sandbox.
Use only an isolated simulation host without sensitive mounts or provider
credentials in the worker environment.

## Eval Set Lockout

Seeds 1-100 are the held-out evaluation set.

- During every Evolutionary Search iteration, use seeds 101-125 only.
- Do not inspect or replay seeds 1-100 while writing or selecting candidates.
- After Stage 1 stops and one program is selected, run Stage 2 on seeds 1-100
  exactly once before returning.

Violation invalidates the benchmark.

## Objective

Improve the existing Fix Loop program with K=8 distinct candidates per
iteration. Save the selected program to:

`$EVOSEARCH_DIR/$TASK/evosearch_best_code.py`

Then evaluate it on seeds 1-100 and write:

`$EVOSEARCH_DIR/$TASK/findings.md`

Always use `.venv-robosuite/bin/python3` for Robosuite commands.

## Robosuite API Contract

Read these files before writing candidates:

- `.claude/robosuite/CLAUDE.md`
- `.claude/robosuite/api-reference.md`
- `.claude/robosuite/fix-loop/skills/grasp.md`
- `.claude/robosuite/fix-loop/skills/localize.md`
- `.claude/robosuite/fix-loop/skills/manipulation.md`
- `.claude/robosuite/fix-loop/skills/transport.md`

Forbidden in candidate/debug code:

`env.handle.env`, `env.handle.env.sim`, `sim.data.*`, `sim.model.*`,
`sim.forward()`, `env._step_once()`, and reading simulator XML/URDF/assets.

Use only the perception, geometry, IK, motion, and gripper APIs documented in
the suite constitution. Single-arm and bimanual APIs are different; do not use
unsuffixed motion/gripper calls on bimanual tasks.

All tasks use:

`obs["robot0_robotview"]["images"]["rgb"]`
`obs["robot0_robotview"]["images"]["depth"]`
`obs["robot0_robotview"]["intrinsics"]`
`obs["robot0_robotview"]["pose_mat"]`

## Required Services

Before any replay:

```bash
for p in 8114 8115 8116 8122; do
  echo "port $p: $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:$p/health)"
done
```

All four must respond (`404` is expected; `000` is down). Molmo is required
because both tracked baselines use it as a localization fallback.

## Stage 0 — Resume and Baseline

Run Stage 0 and the later shell snippets in the same persistent Bash shell.
Initialize the shell variables from the filled assignment above:

```bash
TASK="<filled TASK>"
CONFIG="<filled CONFIG>"
FIX_CODE="<filled FIX_CODE>"
GPU="<filled GPU>"
BASELINE_RATE="<filled BASELINE_RATE>"
EVOSEARCH_DIR="outputs/robosuite_evosearch"
EVALDIR="outputs/robosuite_evosearch_eval"
BEST_CODE="$EVOSEARCH_DIR/$TASK/evosearch_best_code.py"
mkdir -p "/tmp/robosuite_evosearch_${TASK}"

test -f "$CONFIG" || { echo "Missing config: $CONFIG"; exit 2; }
test -f "$FIX_CODE" || { echo "Missing Fix Loop code: $FIX_CODE"; exit 2; }
```

If `BEST_CODE` exists, do not start a new search. Validate the exact Stage 2
seed set before deciding whether to report or resume:

```bash
existing_seed_list=$(find "$EVALDIR/$TASK" -maxdepth 1 -type d -name 'trial_*' 2>/dev/null \
  | sed -nE 's#.*trial_([0-9]+)_.*#\1#p' | awk '{print $1 + 0}' \
  | sort -nu | paste -sd' ' -)
expected_seed_list=$(seq 1 100 | paste -sd' ' -)
```

If the lists match, report the existing result. Otherwise, if `BEST_CODE`
exists, resume Stage 2 with that code; if it does not, begin Stage 1.

## Stage 1 — Evolutionary Search on Seeds 101-125

### 1. Create one immutable run directory

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$EVOSEARCH_DIR/$TASK/$RUN_ID"
mkdir -p "$RUN_DIR"
```

Read representative Fix Loop `trace.json`, `summary.txt`, and JPEG keyframes
from seeds 101-125 only. Write `$RUN_DIR/task_analysis.md` with:

1. object and goal geometry inferred from observations;
2. baseline strategy and failure modes;
3. grasp/manipulation/transport constraints;
4. eight distinct hypotheses for `iter_00`;
5. an iteration log.

Never infer geometry from simulator assets or internal state.

### 2. Write K=8 candidates

Create:

`$RUN_DIR/iter_00/candidate_A/code.py` through
`$RUN_DIR/iter_00/candidate_H/code.py`.

Candidate A must be a verbatim copy of `$FIX_CODE` in `iter_00`. Every other
candidate must test a distinct mechanistic hypothesis. Each file begins with a
docstring containing:

- hypothesis;
- how it differs from its parent/baseline;
- expected trace evidence if the hypothesis is wrong;
- parent candidate and iteration, if any.

Do not create eight small parameter variations. Across K=8, vary structural
choices such as localization fallback, mask selection, grasp family, retry
logic, waypoint geometry, bimanual coordination, or contact strategy.

### 3. Evaluate every candidate on the same 25 development seeds

For each `ITER_DIR=$RUN_DIR/iter_NN`, run the following function in one Bash
shell. It resumes seeds already present and uses at most ten parallel replay
workers on the assigned GPU:

```bash
eval_candidate() {
  candidate_dir="$1"
  eval_dir="$candidate_dir/eval"
  log_dir="$candidate_dir/logs"
  mkdir -p "$eval_dir" "$log_dir"

  for seed in $(seq 101 125); do
    padded=$(printf '%02d' "$seed")
    if find "$eval_dir" -maxdepth 1 -type d -name "trial_${padded}_*" | grep -q .; then
      continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge 10 ]; do wait -n || true; done
    (
      MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      .venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
        --args.config "$CONFIG" \
        --args.trial "$seed" \
        --args.replay-code "$candidate_dir/code.py" \
        --args.output-dir "$eval_dir" \
        --args.flat-output \
        --args.no-record-video \
        > "$log_dir/seed_${seed}.log" 2>&1
    ) &
  done
  wait
}

for candidate_dir in "$ITER_DIR"/candidate_*; do
  eval_candidate "$candidate_dir"
done
```

Aggregate the iteration:

```bash
.venv-robosuite/bin/python3 - "$ITER_DIR" "$TASK" <<'PY'
import json
import re
import sys
from pathlib import Path

iter_dir = Path(sys.argv[1])
task = sys.argv[2]
summaries = []

for candidate_dir in sorted(iter_dir.glob("candidate_*")):
    results = []
    for seed in range(101, 126):
        matches = sorted((candidate_dir / "eval").glob(f"trial_{seed:02d}_*"))
        if not matches:
            results.append({"trial": seed, "reward": 0.0, "task_completed": 0, "missing": True})
            continue
        name = matches[-1].name
        reward_match = re.search(r"reward_([0-9.]+)", name)
        done_match = re.search(r"taskcompleted_([01])", name)
        sandbox_match = re.search(r"sandboxrc_(-?[0-9]+)", name)
        results.append({
            "trial": seed,
            "reward": float(reward_match.group(1)) if reward_match else 0.0,
            "task_completed": int(done_match.group(1)) if done_match else 0,
            "sandbox_rc": int(sandbox_match.group(1)) if sandbox_match else 1,
        })
    passed = sum(r["task_completed"] for r in results)
    summary = {
        "candidate": candidate_dir.name,
        "code_path": str(candidate_dir / "code.py"),
        "trials": 25,
        "pass_count": passed,
        "pass_rate": passed / 25,
        "errors": sum(r.get("sandbox_rc", 1) != 0 for r in results),
        "trial_results": results,
    }
    (candidate_dir / "eval_results.json").write_text(json.dumps(summary, indent=2))
    summaries.append(summary)

if len(summaries) != 8:
    raise SystemExit(f"incomplete iteration: expected 8 candidates, found {len(summaries)}")
missing = [
    (item["candidate"], result["trial"])
    for item in summaries
    for result in item["trial_results"]
    if result.get("missing")
]
if missing:
    raise SystemExit(f"incomplete iteration: missing {len(missing)} trials: {missing}")

summaries.sort(key=lambda item: (-item["pass_rate"], item["candidate"]))
combined = {
    "suite": "robosuite",
    "task": task,
    "trial_seeds": list(range(101, 126)),
    "candidates": summaries,
    "best_candidate": summaries[0]["candidate"],
    "best_pass_rate": summaries[0]["pass_rate"],
}
(iter_dir / "iter_summary.json").write_text(json.dumps(combined, indent=2))
for item in summaries:
    print(f"{item['candidate']}: {item['pass_count']}/25 errors={item['errors']}")
PY
```

Nonzero sandbox return codes count as failures. Aggregation exits without an
`iter_summary.json` if any of the K=8 candidates lacks one of the 25 results.

### 4. Diagnose and iterate

Read every leaderboard, then inspect failure traces and keyframes for the
top-three candidates plus any candidate with a unique partial success. Update
`task_analysis.md` before writing the next iteration.

For `iter_01` through `iter_04`:

1. seed candidates from the top-three survivors across all prior iterations;
2. preserve at least one structurally novel candidate;
3. use the same seeds 101-125;
4. do not repeat eliminated hypotheses without new evidence;
5. record leaderboard, eliminated hypotheses, and open questions.

Stop Stage 1 at 25/25 or after five complete iterations. Do not stop early for
a plateau.

### 5. Select and save the best program

Select the highest pass rate across all `iter_summary.json` files. Break ties
by preferring the shorter program as a deterministic simplicity proxy.
Evolutionary Search must not replace the Fix Loop baseline with a worse
development result; fall back to `iter_00/candidate_A/code.py` when necessary.

```bash
SELECTED_CODE=$(.venv-robosuite/bin/python3 - "$RUN_DIR" <<PY
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
ranked = []
for summary_path in sorted(run_dir.glob("iter_*/iter_summary.json")):
    data = json.loads(summary_path.read_text())
    for candidate in data["candidates"]:
        code_path = Path(candidate["code_path"])
        if not code_path.is_file():
            raise SystemExit(f"missing candidate code: {code_path}")
        code_lines = sum(
            bool(line.strip()) and not line.lstrip().startswith("#")
            for line in code_path.read_text().splitlines()
        )
        ranked.append((
            -candidate["pass_rate"], code_lines, summary_path.parent.name,
            candidate["candidate"], str(code_path),
        ))

if not ranked:
    raise SystemExit("no complete iteration summaries found")
print(min(ranked)[-1])
PY
)
test -f "$SELECTED_CODE" || { echo "Missing selected code: $SELECTED_CODE"; exit 2; }
mkdir -p "$EVOSEARCH_DIR/$TASK" outputs/working_codes
cp "$SELECTED_CODE" "$EVOSEARCH_DIR/$TASK/evosearch_best_code.py"
cp "$SELECTED_CODE" "outputs/working_codes/robosuite_${TASK}_evosearch.py"
```

Record the selected path, iteration, candidate, and 25-seed rate in
`task_analysis.md`.

## Stage 2 — Selected Program on Seeds 1-100

Run only after Stage 1 selection. Resume existing unique seeds and never run a
second candidate on this partition:

```bash
BEST_CODE="$EVOSEARCH_DIR/$TASK/evosearch_best_code.py"
STAGE2_DIR="$EVALDIR/$TASK"
STAGE2_LOGS="/tmp/robosuite_evosearch_${TASK}/stage2_logs"
mkdir -p "$STAGE2_DIR" "$STAGE2_LOGS"

for seed in $(seq 1 100); do
  padded=$(printf '%02d' "$seed")
  if find "$STAGE2_DIR" -maxdepth 1 -type d -name "trial_${padded}_*" | grep -q .; then
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge 10 ]; do wait -n || true; done
  (
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    .venv-robosuite/bin/python3 scripts/robosuite/replay_trial_robosuite.py \
      --args.config "$CONFIG" \
      --args.trial "$seed" \
      --args.replay-code "$BEST_CODE" \
      --args.output-dir "$STAGE2_DIR" \
      --args.flat-output \
      --args.no-record-video \
      > "$STAGE2_LOGS/seed_${seed}.log" 2>&1
  ) &
done
wait
```

Count unique trials and successes:

```bash
stage2_seed_file="/tmp/robosuite_evosearch_${TASK}/stage2_seeds.txt"
find "$STAGE2_DIR" -maxdepth 1 -type d -name 'trial_*' \
  | sed -nE 's#.*trial_([0-9]+)_.*#\1#p' | awk '{print $1 + 0}' \
  | sort -nu > "$stage2_seed_file"
stage2_total=$(wc -l < "$stage2_seed_file")
stage2_success=$(find "$STAGE2_DIR" -maxdepth 1 -type d -name '*taskcompleted_1' \
  | sed -nE 's#.*trial_([0-9]+)_.*#\1#p' | awk '{print $1 + 0}' \
  | sort -nu | wc -l)
echo "Stage 2: $stage2_success/$stage2_total"
diff -u <(seq 1 100) "$stage2_seed_file"
```

## Findings

Write `$EVOSEARCH_DIR/$TASK/findings.md`:

```text
## Task: <TASK>
## Fix Loop held-out rate: <BASELINE_RATE>
## Best Evolutionary Search development rate: <N>/25
## Stage 2 held-out result: <N>/100

### Evolutionary Search Run
- Run directory:
- Iterations completed:
- Selected candidate:
- Stopping reason: solved|max_iterations

### What Changed vs Fix Loop
- ...

### Failure Modes Eliminated
- ...

### Generalizable Patterns
- ...

### Validity / Anomalies
- seed counts, missing traces, service errors, retries, or "none"
```

Promote only genuinely generalizable findings to the existing Robosuite Fix
Loop skills and append a dated entry under `docs/logs/`.

## Return

Return:

- task/config/GPU;
- selected candidate and run directory;
- Stage 1 result and stopping reason;
- Stage 2 result with confirmation that 100 unique seeds exist;
- three concise key findings;
- every file changed outside runtime output directories.
```
