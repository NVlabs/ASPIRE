"""Stage a prompt package for one worker or one skill verifier.

The coordinator dispatches every agent from a file rather than from an inline prompt, so that what
each agent was told is recoverable afterwards. This script builds that file from the checked-in
template plus the standing addenda below.

The addenda are not style: each line is a hazard that cost a real run during the 80-task campaign,
and they live here rather than in the templates because they are *coordinator* policy — how an
unattended fleet must behave — while the templates describe the job.

    # a Stage 1 worker
    uv run python .../dispatch.py worker --suite libero_goal_swap --task-id 4 --gpu 2

    # a skill verifier: also writes VERIFY_DIR/proposed_skills.md from the task's skill_report
    uv run python .../dispatch.py verify --suite libero_goal_swap --task-id 4 --gpu 2

Prints the path of the prompt file it wrote. Pass it to the agent as "read this file and follow
it exactly" — do not paste the contents inline, or the record is lost when the context is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign import (
    ROOT,
    SKILL_ROOT,
    campaign_settings,
    format_seed_range,
    resolve_run_root,
    verification_dir,
)
from campaign import (
    task_dir as campaign_task_dir,
)

WORKER_TEMPLATE = SKILL_ROOT / "subagent-prompt.md"
VERIFIER_TEMPLATE = SKILL_ROOT / "verifier-prompt.md"
START = "<!-- ==================== TEMPLATE START ==================== -->"
END = "<!-- ==================== TEMPLATE END"

# Shared by both roles. Every item is retroactive.
COMMON_ADDENDA = """
1. **Set your working directory explicitly to `{cap_root}` at the start of every shell command.
   Do NOT trust the inherited cwd.** Use absolute paths everywhere.
2. **Never run `rm`**, in any form, including inside a compound command — and never anything else
   that needs a human to approve it. An unattended campaign does not fail on an approval prompt,
   it *waits*: three agents stalled this way, two of them for over 14 hours each. To set something
   aside, `mv` it into an `archive/` directory inside your own working directory. The same applies
   to `sudo`, to backgrounding (`nohup`, `&`), and to any write outside your assigned directory.
3. **If a tool refuses an action, do NOT reach for a different tool to accomplish the same thing.**
   Report the refusal in your summary and continue. Routing around a refusal is indistinguishable
   from evading a control, and it cost one agent's output its credibility.
4. The sandbox builtin whitelist is narrow: no `getattr`/`hasattr`/`isinstance`/`type`, no name
   beginning with an underscore, a generator is consumed once, and `gripper_positions["primary"]`
   is a scalar float.
"""

WORKER_ADDENDA = """
5. **Probe sweeps must not touch the unseen-check seeds {unseen}.** Range-check every `--seeds`.
6. **If your first program passes every development seed, do NOT stop there.** Spend the unused
   version budget on **one-variable ablations** of whatever your program most depends on, and
   report what they measure. Two tasks in that position produced the campaign's most useful
   findings that way. A pass count frequently cannot tell two placements apart — see
   `transport.md` §*Grade a Placement by the Object's LANDED POSE*.
7. **When you report your unseen result, name the residual failure REGIME, not a predicted rate.**
   Workers that named a specific remaining mode were well calibrated against held-out; workers
   reasoning from the aggregate development rate were wrong by up to 30 points.
"""

VERIFIER_ADDENDA = """
5. **Never read another task's directory**, and never modify anything outside `VERIFY_DIR`. The
   skill library is the only sanctioned cross-task channel; one agent disclosed reading a sibling
   task and its library signal had to be discarded.
6. **`Write` sometimes refuses report-shaped `.md` files in a subagent.** If that happens, return
   the content as text in your summary and say so. Do not write it another way.
"""

TASK_SUITE_NOTE = """
- **This is a `_task` suite: the instruction and the BDDL task name deliberately disagree.**
   `get_task_context().language` is the only authority. The decoy is usually a different OBJECT,
   but one task found it inverting the VERB instead ("turn off" against a name of `turn_on_...`) —
   read the whole instruction, not just its noun.
"""

# Claims the campaign could not settle. A verifier that measures one of these on a fresh scene is
# worth more than one that returns a verdict, so they are handed over verbatim.
STANDING_DISPUTES = """
## Standing disputes — report on any your scene touches

- **Identity selectors.** `localize.md` orders them: the relation the instruction states, else the
  sign of the support gap, else the candidate's own height, else proximity. Tasks measured these
  head to head and got different answers on different scenes. **Compute all of them in-program and
  report the column.**
- **The near-base descent.** Two regimes with opposite remedies — settle-time (more ticks fix it)
  and a servo wall (they do not). The `>= ~50 ticks` duration bar is RETIRED; so is the
  outward-drift test, which one task measured backwards. What survives is **where it froze**
  (achieved z against commanded z) and **which axis is short**. Report both.
- **The jaw as a slip test.** Five numeric bars have been tried and all five failed; a jaw that
  *rises* is a distinct signal (something is prying it). If you deliberately spoil a grasp, report
  the jaw at close / after lift / before place for both it and a good one.
- **Release height.** Measure your band and grade on the landed pose, not the pass count.
- **The first `move_to_pose`.** `tries=3` and stage only if all three fail. Report what yours does.

Verifications have refuted claims outright while their symptoms held, and found helper functions
that entries CALL and that were defined nowhere. Treat "it passed" as the beginning of the question.
"""


def fill(template: Path, replacements: dict[str, str]) -> str:
    body = template.read_text().split(START)[1].split(END)[0]
    for old, new in replacements.items():
        body = body.replace(old, new)
    return body.strip()


def stage_proposals(run_root: Path, benchmark: str, suite: str, task_id: int, out: Path) -> int:
    """Write the verifier's copy of the proposals, and return how many there are.

    These are staged OUTSIDE the task directory and are deliberately NOT applied to the shared
    library: promotion is what the verdict decides, and an unverified entry must not reach the
    tasks running alongside. An earlier version of this header claimed the opposite and several
    verifiers wasted time hunting for entries in `skills/` that were never meant to be there.
    """
    report_path = campaign_task_dir(run_root, benchmark, suite, task_id) / "skill_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"no skill_report.json for {suite}/task_{task_id} — nothing was proposed, so there is "
            "nothing to verify. Skip this task rather than dispatching a verifier."
        )
    report = json.loads(report_path.read_text())
    new = report.get("proposed_new") or []
    edits = report.get("proposed_edits") or []
    if not new and not edits:
        raise SystemExit(f"{suite}/task_{task_id} proposed nothing — no verifier needed.")

    manifests = sorted(
        (campaign_task_dir(run_root, benchmark, suite, task_id) / "validation" / "runs").glob(
            "*/manifest.json"
        )
    )
    passes = json.loads(manifests[-1].read_text())["passes"] if manifests else "?"

    lines = [
        "# Skills Under Test\n",
        f"Proposals from `{suite}/task_{task_id}`. **They are staged here and are deliberately "
        "NOT in the shared library** — whether they get promoted is what your verdict decides. "
        "Do not go looking for them under `skills/`; this file is their only home. Re-solve the "
        "task from the library PLUS this file, without seeing the author's program.\n",
        f"The task's own program scores **{passes}/50** held-out. Read that before drawing "
        "conclusions from your own score: on a task whose residual failure is kinematic, a low "
        "score of your own is expected and is **not** by itself a refutation. Judge each proposal "
        "on whether it transmits and whether its mechanism reproduces.\n",
        "---\n",
    ]
    for i, p in enumerate(new, 1):
        lines.append(f"## {i}. → `{p['skill']}` — {p['title']}\n\n**Trigger.** {p['trigger']}\n")
        lines.append("```python\n" + (p.get("code") or "").strip() + "\n```\n")
        lines.append(f"**Evidence.** {p.get('evidence', '')}\n\n---\n")
    if edits:
        lines.append("## Edits\n")
        for e in edits:
            lines.append(f"### `{e['skill']}` § {e.get('section', '')}\n")
            lines.append(f"**Change.** {e.get('change', '')}\n\n*Why:* {e.get('why', '')}\n")
    lines.append("\n---\n" + STANDING_DISPUTES)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return len(new) + len(edits)


def lint_staged_proposal(proposal: Path, skip: bool = False) -> None:
    """Run lint_skills.py over the staged proposal before a GPU is spent on it.

    A verifier costs a GPU slot and ~15 minutes; a helper that no fence defines costs it the
    whole run. Three verifiers in one campaign each opened with `NameError: quantile`, and a
    fourth had to reconstruct a snippet from prose because the proposal pointed into the
    proposer's own directory. Every one of those is an `ast` walk, not an experiment.

    Advisory by default in one direction only: a lint failure blocks the dispatch, because the
    cheapest moment to fix a proposal is before anyone reads it. `--skip-lint` is the escape
    hatch for a proposal whose fences are deliberately partial.
    """
    linter = Path(__file__).resolve().parent / "lint_skills.py"
    result = subprocess.run(
        [sys.executable, str(linter), "--proposal", str(proposal)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("lint: clean", file=sys.stderr)
        return
    print(result.stdout, file=sys.stderr)
    if skip:
        print("lint: FAILED (dispatching anyway: --skip-lint)", file=sys.stderr)
        return
    raise SystemExit(
        f"lint failed on {proposal}. Fix the proposal, or re-run with --skip-lint if the fences "
        "are deliberately partial. Do not spend a verifier on a NameError."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("worker", "verify"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--benchmark", default="libero-pro")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument(
        "--unseen-seeds",
        default=None,
        help="seeds a worker must never probe; range-checked in the addenda "
        "(default: unseen_seeds from campaign.json, e.g. 66-80 for LIBERO, 126-130 for Robosuite)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the prompt (default: RUN_ROOT/prompts)",
    )
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        help="file whose contents are appended verbatim (task-specific notes)",
    )
    parser.add_argument(
        "--skip-lint",
        action="store_true",
        help="stage a verifier package even if the proposal fails lint_skills.py",
    )
    args = parser.parse_args()

    run_root = resolve_run_root(args.run_root)
    out_dir = args.out_dir or run_root / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = COMMON_ADDENDA.format(cap_root=ROOT)
    unseen_seeds = args.unseen_seeds or format_seed_range(
        campaign_settings(run_root)["unseen_seeds"]
    )

    if args.role == "worker":
        body = fill(
            WORKER_TEMPLATE,
            {
                "BENCHMARK: <libero-pro|robosuite|robocasa>": f"BENCHMARK: {args.benchmark}",
                "SUITE:     <full suite name, e.g. libero_goal_swap>": f"SUITE:     {args.suite}",
                "TASK_ID:   <integer task id>": f"TASK_ID:   {args.task_id}",
                "GPU:       <assigned simulation GPU>": f"GPU:       {args.gpu}",
                "RUN_ROOT:  <campaign root, e.g. outputs/runs/2026-07-09_14-32-00>": f"RUN_ROOT:  {run_root}",
            },
        )
        addenda = common + WORKER_ADDENDA.format(unseen=unseen_seeds)
        out = out_dir / f"worker_{args.suite}_{args.task_id}.txt"
    else:
        verify_dir = verification_dir(run_root, args.benchmark, args.suite, args.task_id)
        count = stage_proposals(
            run_root, args.benchmark, args.suite, args.task_id, verify_dir / "proposed_skills.md"
        )
        body = fill(
            VERIFIER_TEMPLATE,
            {
                "`<benchmark>`": f"`{args.benchmark}`",
                "`<full suite name>`": f"`{args.suite}`",
                "`<task id>`": f"`{args.task_id}`",
                "`<N>`": f"`{args.gpu}`",
                "`<campaign root>/verification/<benchmark>__<suite>__task_<task id>`": f"`{verify_dir}`",
            },
        )
        addenda = common + VERIFIER_ADDENDA
        out = out_dir / f"verify_{args.suite}_{args.task_id}.txt"
        print(f"staged {count} proposal(s) -> {verify_dir / 'proposed_skills.md'}", file=sys.stderr)
        lint_staged_proposal(verify_dir / "proposed_skills.md", skip=args.skip_lint)

    text = body + "\n\n---\n\n## Coordinator addenda\n" + addenda
    if args.suite.endswith("_task"):
        text += TASK_SUITE_NOTE
    if args.append is not None:
        text += "\n" + args.append.read_text()
    out.write_text(text)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
