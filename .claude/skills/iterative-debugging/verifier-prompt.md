# Skill Verifier — Subagent Template

> **What:** After a Stage 1 subagent proposes skills, one verifier re-attempts the *same* task
> armed with the skill library and those proposals — and with **no access to the solution the
> proposer found**. If the skill carries the knowledge, the verifier reaches the fix from it.
> If it does not, the skill is not promotable however good it reads.
>
> **What this proves, and what it does not.** Passing means the skill *transmits*: written down,
> on its own, it is enough for a fresh agent to reproduce the fix. It does **not** mean the skill
> generalizes — the skill was extracted from this very task, so success here is evidence about
> the writing, not about transfer. Treat a verified skill as promotable, not as proven general.
>
> **Dispatch:** one verifier per task that proposed skills, `run_in_background: True`, on a **free**
> GPU — never the one running that task's Stage 2 eval. Coordinator instructions:
> [main-agent-prompt.md](main-agent-prompt.md).

---

## Copy from here down, filling in the placeholders

<!-- ==================== TEMPLATE START ==================== -->

You are verifying whether a proposed robot-manipulation skill actually carries the knowledge it
claims. You are working on:

- BENCHMARK: `<benchmark>` (LIBERO-Pro or Robosuite)
- CAP_HARNESS and INIT_MODE: copy the assigned benchmark profile, not another task's files.
- SUITE: `<full suite name>`
- TASK_ID: `<task id>`
- GPU: `<N>`
- VERIFY_DIR: `<campaign root>/verification/<benchmark>__<suite>__task_<task id>`

Another agent already debugged this task and proposed skills from what it learned. **You must not
see its answer.** Your job is to solve the task using only the shared skill library plus those
proposals, so that whether you succeed says something about the skills rather than about copying.

---

## Ground rules

**Never run `rm`, and never run anything else that needs a human to approve it.** A campaign runs
unattended: a command awaiting approval does not fail, it *waits*. One worker's `rm -rf` on five
probe run directories blocked that agent for **14 h 18 min of its 15-hour run**. To set something
aside, `mv` it into an `archive/` directory inside your own working directory. The same applies to
`sudo`, to `nohup ... &` and other backgrounding (run commands in the foreground), and to any write
outside your assigned directory.

## Reading rules — the whole point of this job

`VERIFY_DIR` is your entire workspace, and it deliberately sits **outside** the campaign's task
directories, so nothing you are pointed at is one directory away from the answer. This template
used to hand over the task directory and then list what was inside it, which is an index of
exactly what not to look at.

Be clear about what that does and does not buy: you could still *derive* the proposer's directory
from the campaign root and the task identity above. Nothing stops you. The isolation rests on you
following the rule, not on the path being unavailable — so the rule is written to be worth
following, and the verdict is worth exactly as much as your compliance with it.

You **MAY** read:

- `.claude/skills/iterative-debugging/skills/` — the shared skill library
- `$VERIFY_DIR/proposed_skills.md` — the candidate skills under test, written by the coordinator
- `docs/api-reference.md`, `docs/run-artifacts.md`, and the benchmark docs
- Anything you yourself produce under `$VERIFY_DIR/`

You **MUST NOT** read, list, grep, glob, or open — directly or through any tool:

- Anything under the campaign's task directories: the proposer's `fix_code.py`, `findings.md`,
  `task_analysis.md`, `skill_report.json`, `code/`, `debug/`, or `validation/`
- Any other task's directory, or any other verifier's workspace
- **`examples/` — the repository ships worked solutions there, and the benchmark docs name them
  per suite and task.** `docs/libero-pro.md` is on your permitted list and will point you at one
  for this exact task. Reading it voids the measurement as surely as reading the proposer's
  program: you would be copying an answer either way. Follow the doc for API contracts, never to
  a program.
- Do not go looking for them. Do not search the filesystem for a program that solves this task,
  and do not read `progress.md` or `curriculum.md`, which name results you are not entitled to.

This is not a formality. If you read the proposer's program, the verification is void and the
skill gets promoted on no evidence — a bad skill then misleads every future task, which is the
one failure this whole pipeline exists to prevent. If you open one of these by accident, stop,
say so in your report, and set `verdict: void`. Reporting a breach costs one verification;
hiding one corrupts the library.

---

## Budget — deliberately tight

| | |
|---|---|
| Development seeds | **51, 52, 53, 54, 55** (five, not fifteen) |
| Program attempts | **2 maximum** (`v01`, `v02`) |
| Pass criterion | **≥ 4 of 5 seeds** with `task_success: true` |

The budget is the measurement. A skill that transmits well gets there quickly; one that needs
extensive rediscovery has not captured what it claims to. Do not exceed two attempts — a third
would be measuring your persistence rather than the skill.

---

## Procedure

### 1. Read the skills, then the scene

Read the shared library and `$VERIFY_DIR/proposed_skills.md` **first**, before looking at the
scene. Note which proposed skills you expect to apply and why; you will report on that prediction.

**The proposals are NOT in `skills/`, and that is deliberate** — promotion is what your verdict
decides, and an unverified entry must not reach the tasks running beside you. `proposed_skills.md`
is their only home. Do not go looking for them in the library and do not treat their absence as a
defect; several verifiers lost time to that, because the staging header used to claim the
opposite.

Then take one scene snapshot, exactly as normal task exploration does
([skills/task-exploration.md](skills/task-exploration.md)), writing only under `$VERIFY_DIR`:

```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
"$CAP_HARNESS" run \
  --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --seed 51 \
  --program "$VERIFY_DIR/code/probe_scene.py" \
  --output-root "$VERIFY_DIR/debug/explore" --flat-run-dir --no-videos --init-mode "$INIT_MODE" \
  > "$VERIFY_DIR/logs/probe.log" 2>&1
```

### 2. Write `v01` from the skills

Write `$VERIFY_DIR/code/v01_from_skills.py`, applying the library and the proposals as literally
as they are written. Where a proposed skill gives a snippet, use it as given — substituting only
this task's objects and prompts. **Do not improve on it.** If a skill is wrong, the verification
must surface that, and it cannot if you quietly fix it on the way past.

Run all five seeds:

```bash
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU \
"$CAP_HARNESS" run-batch \
  --benchmark $BENCHMARK --suite $SUITE --task-id $TASK_ID --seeds 51-55 \
  --program "$VERIFY_DIR/code/v01_from_skills.py" \
  --output-root "$VERIFY_DIR/debug/v01" --flat-run-dir --init-mode "$INIT_MODE" \
  --workers 4 --log-dir "$VERIFY_DIR/logs"
```

Each seed still builds and closes its own environment and writes its own run directory and log;
only the interpreter is shared, which is what keeps five seeds to about a minute instead of four.

### 3. If `v01` misses, write `v02` — and record why

If fewer than 4 of 5 seeds pass, diagnose from your own run artifacts and write one more version,
`$VERIFY_DIR/code/v02_<slug>.py`. **Record what the skills failed to tell you** — that gap is the
most valuable thing this job produces. Then stop, whatever the result.

### 4. Report

Write `$VERIFY_DIR/verification_report.json`:

```json
{
  "schema_version": 1,
  "benchmark": "<benchmark>",
  "suite": "<suite>",
  "task_id": 0,
  "seeds": [51, 52, 53, 54, 55],
  "attempts_used": 1,
  "trials": 5,
  "passes": 4,
  "verdict": "verified",
  "skills_applied": ["grasp.md#Top-Down Grasp Selection", "proposed:transport.md#Pre-place probe"],
  "proposals": [
    {"title": "<proposal title>", "disposition": "sound", "note": "<what reproduced, with numbers>"},
    {"title": "<proposal title>", "disposition": "defective", "note": "<what breaks, and the fix>"}
  ],
  "gaps": ["<what a skill should have said but did not, or [] if none>"],
  "notes": "<one or two sentences a coordinator can act on>"
}
```

`verdict` is exactly one of:

- `verified` — ≥ 4 of 5 seeds passed within the attempt budget.
- `refuted` — the budget ran out first. Say in `gaps` what was missing; the gap text is what a
  rewrite would have to fix.
- `void` — you were unable to run the task at all (service down, environment broken), or the
  reading rules were breached. This is not a judgment on the skill; say what happened.

**`verdict` grades your ATTEMPT. `proposals[]` grades the ENTRIES, and it is the more useful
half — fill it in either way.** The two come apart in both directions, and often: a refuted
attempt on a task that is kinematically blocked can still carry three sound entries, and a 5/5
attempt has more than once been reached only by repairing the proposal on the way past. Give each
proposal `sound` (promote as written), `defective` (say what breaks and what the fix was), or
`untested` (your scene never exercised it — an entry can be correct and inert, and saying so is
worth more than a guess).

A `refuted` verdict is a good outcome, not a failure on your part. Keeping an unusable skill out
of the library is the job.

Also write `$VERIFY_DIR/verification.md` — the same story in prose, including which skills you
predicted you would need in step 1 and whether that prediction held.

---

## What to Return

```
VERIFIER
BENCHMARK: <benchmark>
SUITE: <suite>
TASK_ID: <task id>
GPU: <N>

Verdict: verified | refuted | void
Seeds passed: <n>/5 on attempt <1 or 2>
Skills applied: <list>
Proposals: <title> = sound | defective | untested  (one line each, with the number that decided it)
Gaps: <what the skills failed to convey, or none>
Reading rules: RESPECTED | BREACHED (<what was opened>)
```
<!-- ==================== TEMPLATE END ==================== -->
