#!/usr/bin/env python3
"""Auto-generate the iterative-debugging progress file from filesystem state.

Run: python3 .claude/skills/iterative-debugging/scripts/gen_progress.py
     (or append --print to stdout only, without writing the file)

The coordinator should run this instead of manually editing the progress file.
It targets the campaign at ``outputs/runs/LATEST`` unless ``--run-root`` is
given. The task list comes from ``<run-root>/tasks.json`` (written once by
``init_run.py``); status is derived from disk artifacts, not agent memory.

tasks.json format — a JSON list of task entries:

    [
      {"benchmark": "libero-pro", "suite": "libero_goal_swap", "task_id": 0,
       "task_name": "open_the_middle_drawer_of_the_cabinet"},
      ...
    ]

``task_name`` is optional display metadata; identity is (benchmark, suite, task_id).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign import (
    heldout_seeds,
    read_json,
    resolve_run_root,
    skill_library_sha,
    validate_skill_report,
    verification_dir,
)
from campaign import task_dir as campaign_task_dir

# Historical defaults; main() rebinds both from the campaign's campaign.json
# (init_run.py --heldout-count) so a 20-seed campaign can reach "done".
HELDOUT_SEEDS = [str(seed) for seed in range(1, 51)]
DONE_THRESHOLD = 50  # every task must complete the full held-out partition
#: Below this held-out pass rate a task is a candidate for the end-of-campaign
#: rerun sweep -- but only once the library it ran under is no longer current.
RERUN_PASS_RATE = 0.8


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(tasks_path: Path) -> list[dict]:
    if not tasks_path.is_file():
        raise SystemExit(
            f"task list not found: {tasks_path}\n"
            "Write it once at campaign start (see this script's docstring for the format)."
        )
    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"task list must be a non-empty JSON list: {tasks_path}")
    for entry in tasks:
        for field in ("benchmark", "suite", "task_id"):
            if field not in entry:
                raise SystemExit(f"task entry missing '{field}': {entry}")
    return tasks


def task_dir_name(task_id: int) -> str:
    return f"task_{task_id}"


def latest_validation_manifest(
    task_eval_dir: Path, *, code_sha256: str | None = None
) -> dict | None:
    """Return the best coherent manifest matching the selected code.

    Full 1-50 runs outrank subset diagnostics; recency breaks ties.
    """
    candidates = []
    # rglob, not glob: a phase-3 rerun's Stage 2 runs with an explicit
    # --output-root, and run_validation then nests the manifest under an extra
    # <benchmark>/<suite>/task_<id>/ level. A fixed-depth glob never matched it,
    # so a rerun scored zero trials, flipped the task from done back to
    # stage1-done, and the campaign's completion gate never closed.
    for path in task_eval_dir.rglob("manifest.json"):
        if path.parent.parent.name != "runs":
            continue
        try:
            manifest = json.loads(path.read_text())
            identity = manifest.get("identity", {})
            if code_sha256 and identity.get("code_sha256") != code_sha256:
                continue
            seeds = identity.get("seeds", [])
            is_full = [str(seed) for seed in seeds] == HELDOUT_SEEDS
            updated = manifest.get("updated_at", "")
            candidates.append((is_full, updated, path.stat().st_mtime, path, manifest))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        return None
    _, _, _, path, manifest = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    manifest["_manifest_path"] = str(path)
    return manifest


def manifest_counts(manifest: dict) -> tuple[int, int]:
    """Count held-out trials and passes from one immutable run only."""
    rows = manifest.get("results", {})
    valid = [row for key, row in rows.items() if str(key) in HELDOUT_SEEDS]
    # A pass is task_success alone (matching the original pipeline's
    # taskcompleted flag); program_ok is a triage signal, not a gate.
    return len(valid), sum(int(bool(row.get("task_success"))) for row in valid)


def get_status(fix_code: Path | None, trials: int) -> str:
    """Task state from disk artifacts alone.

    ``stage1-running`` exists because a worker writes ``fix_code.py`` at Step 3
    and ``skill_report.json`` at Step 6, so between those two there is a window,
    minutes long, where the task looks finished and its report looks lost. Both
    readings are actionable and both are wrong: the coordinator would start a
    50-seed eval on a GPU another job still owns, and would write off findings
    that simply are not written yet. Disk cannot tell "still running" from "died
    after Step 3" -- only the coordinator's ledger can -- so this reports the
    ambiguity instead of guessing, and keeps the task out of both work queues
    until it resolves.

    An *invalid* report counts as still-running for the same reason a missing one
    does. A worker that is mid-Step-6 writes the file, runs the validator, and
    rewrites it until it passes -- so a report that does not validate is most
    often a snapshot of that loop, not a finished task. Reading it as
    ``stage1-done`` put a task into the Stage 2 queue while its worker was still
    live on that GPU, which is the one collision the ledger exists to prevent.
    """
    if fix_code is None or not fix_code.exists():
        return "pending"
    if trials >= DONE_THRESHOLD:
        return "done"
    report_path = fix_code.parent / "skill_report.json"
    if not report_path.is_file():
        return "stage1-running"
    report = read_json(report_path)
    if report is None or validate_skill_report(report, check_skill_files=False):
        return "stage1-running"
    return "stage1-done"


def attempts(task_dir: Path) -> list[Path]:
    """Every attempt at a task, oldest first: the original, then ``rerun_NN/``.

    Phase 3 reruns a weak task against the final library and writes to a fresh
    directory so the first attempt's evidence is never overwritten. Scoring has
    to follow, or the rerun is invisible: progress keeps reporting the original
    result, and -- because the stale attempt still carries the old library hash
    -- the task is re-listed as a rerun candidate on every regeneration.
    """
    found = [task_dir] if (task_dir / "fix_code.py").is_file() else []
    found.extend(
        sorted(
            (path for path in task_dir.glob("rerun_*") if (path / "fix_code.py").is_file()),
            key=lambda path: path.name,
        )
    )
    return found


def latest_attempt(task_dir: Path) -> tuple[Path, str | None]:
    """The attempt a task is currently judged by, and its label if it is a rerun."""
    found = attempts(task_dir)
    if not found:
        return task_dir, None
    newest = found[-1]
    return newest, (None if newest == task_dir else newest.name)


def attempt_with_skills(task_dir: Path) -> Path:
    """The attempt whose skill report the queue should reflect.

    Newest first, but skipping attempts that have not written one yet: the
    moment a rerun wrote `fix_code.py`, judging skills by the newest attempt
    alone dropped the original attempt's still-unverified proposals out of the
    verification queue without ever verifying or refuting them.
    """
    for attempt in reversed(attempts(task_dir) or [task_dir]):
        if (attempt / "skill_report.json").is_file():
            return attempt
    return task_dir


def skill_state(attempt_dir: Path, verify_dir: Path) -> tuple[str, int]:
    """Where this task's proposed skills stand, and how many it proposed.

    The proposal comes from the attempt that made it; the verdict comes from a
    workspace outside the task tree, because a verifier that can see the task
    directory can see the answer it is supposed to rediscover.

    Skill verification is deliberately *not* part of the task lifecycle: it
    gates promotion into the shared library, not the task's own held-out
    evaluation, so the two proceed independently.

    Returns one of:
      missing   -- no report at all; Step 6 is required, so this is a gap
      invalid   -- a report that cannot be trusted (see validate_skill_report)
      none      -- a valid report proposing nothing, which is a real result
      proposed  -- proposals are waiting for a verifier
      verified  -- a verifier reproduced the fix from the proposals alone
      refuted   -- a verifier could not, so the proposals are not promotable

    `missing` and `invalid` are distinct from `none` on purpose. Collapsing
    them let a malformed report read exactly like a task that had nothing to
    say, so the one case a coordinator must act on was the one it could not see.
    """
    if not (attempt_dir / "skill_report.json").is_file():
        return "missing", 0
    report = read_json(attempt_dir / "skill_report.json")
    # Structural checks only. Whether a graded skill file still exists is a
    # question about today's library, not about this report: deleting a skill
    # graded `wrong` -- which rule 6a instructs -- would otherwise mark every
    # earlier task that graded it `invalid` and drop its pending proposals.
    if report is None or validate_skill_report(report, check_skill_files=False):
        return "invalid", 0
    proposals = list(report.get("proposed_new") or []) + list(report.get("proposed_edits") or [])
    if not proposals:
        return "none", 0
    verification = read_json(verify_dir / "verification_report.json")
    if verification is None:
        return "proposed", len(proposals)
    verdict = str(verification.get("verdict", "")).lower()
    return (verdict if verdict in {"verified", "refuted"} else "proposed"), len(proposals)


def fmt_rate(passes: int, trials: int) -> str:
    if trials == 0:
        return "—"
    pct = passes * 100 // trials
    return f"{passes}/{trials} ({pct}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="campaign root (default: outputs/runs/LATEST)",
    )
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--print", dest="print_only", action="store_true")
    parser.add_argument(
        "--rerun-threshold",
        type=float,
        default=RERUN_PASS_RATE,
        metavar="RATE",
        help=(
            "held-out pass rate below which a done task joins the end-of-campaign "
            f"rerun sweep (default: {RERUN_PASS_RATE})"
        ),
    )
    args = parser.parse_args()

    run_root = resolve_run_root(args.run_root)
    out_file = run_root / "progress.md"
    tasks = load_tasks(args.tasks if args.tasks is not None else run_root / "tasks.json")
    global HELDOUT_SEEDS, DONE_THRESHOLD
    heldout = heldout_seeds(run_root)
    HELDOUT_SEEDS = [str(seed) for seed in heldout]
    DONE_THRESHOLD = len(heldout)
    current_library = skill_library_sha()

    lines = []
    lines.append("# Iterative Debugging Progress")
    lines.append("")
    lines.append(
        "*Auto-generated by `.claude/skills/iterative-debugging/scripts/gen_progress.py`"
        " — do not edit manually.*"
    )
    lines.append(
        f"**Status key:** `done` = {DONE_THRESHOLD} coherent held-out trials (seeds 1–"
        f"{DONE_THRESHOLD}) | `stage1-done` selected fix"
        " awaiting or continuing validation | `stage1-running` worker not finished |"
        " `pending` Stage 1 and final code selection required"
    )
    lines.append(
        "**Skills key:** `proposed` = awaiting a verifier | `verified` = a verifier reproduced"
        " the fix from the skill alone, promotable | `refuted` = it could not, do not promote |"
        " `invalid`/`missing` = the report cannot be graded, and the task's findings are stranded"
    )
    lines.append(f"**Skill library:** `{current_library}` (content hash of `skills/`)")
    lines.append("")

    pending_tasks = []
    stage1_done_tasks = []
    stage1_running_tasks = []
    verify_tasks = []
    unusable_reports = []
    rerun_candidates = []
    group_rows: dict[tuple[str, str], list] = {}

    for entry in tasks:
        benchmark = entry["benchmark"]
        suite = entry["suite"]
        task_id = int(entry["task_id"])
        label = entry.get("task_name") or task_dir_name(task_id)
        ref = f"{benchmark}/{suite}/{task_dir_name(task_id)}"

        task_dir = campaign_task_dir(run_root, benchmark, suite, task_id)
        # Judge a task by its newest attempt. A phase-3 rerun writes beside the
        # original rather than over it, so scoring the original forever would
        # hide the rerun and re-list the task as a candidate every time.
        attempt, rerun_label = latest_attempt(task_dir)
        fix_code = attempt / "fix_code.py"
        fix_code = fix_code if fix_code.exists() else None
        eval_dir = attempt / "validation"
        manifest = latest_validation_manifest(
            eval_dir, code_sha256=sha256_file(fix_code) if fix_code else None
        )
        if manifest is not None:
            trials, passes = manifest_counts(manifest)
            run_note = f" [run {manifest.get('run_id', 'unknown')}]"
        else:
            # Manifests may exist but none match the selected code.
            trials, passes, run_note = 0, 0, ""
        status = get_status(fix_code, trials)
        rate = (fmt_rate(passes, trials) + run_note) if trials > 0 else "—"
        if rerun_label:
            label = f"{label} [{rerun_label}]"
        skills_attempt = attempt_with_skills(task_dir)
        skills_label = None if skills_attempt == task_dir else skills_attempt.name
        verify_dir = verification_dir(run_root, benchmark, suite, task_id, skills_label)
        skills, proposal_count = skill_state(skills_attempt, verify_dir)
        skill_cell = f"{skills} ({proposal_count})" if proposal_count else skills
        group_rows.setdefault((benchmark, suite), []).append(
            (task_id, label, status, rate, skill_cell)
        )

        if status == "pending":
            pending_tasks.append(ref)
        elif status == "stage1-done":
            stage1_done_tasks.append(ref)
        elif status == "stage1-running":
            stage1_running_tasks.append(ref)
        if skills == "proposed":
            verify_tasks.append(f"{ref} ({proposal_count} proposal(s))")
        elif skills in ("missing", "invalid") and status not in ("pending", "stage1-running"):
            unusable_reports.append(
                f"{ref} — skill_report.json is {skills}; run "
                "`campaign.py --validate-skill-report` on it"
            )
        if status == "done" and trials:
            # A rerun is only worth GPU time when the library actually moved on
            # since this task ran; a weak result under the *current* library
            # would just be reproduced.
            ran_under = (read_json(attempt / "skill_report.json") or {}).get("skill_library_sha")
            if passes / trials < args.rerun_threshold and ran_under != current_library:
                rerun_candidates.append(
                    f"{ref} — {fmt_rate(passes, trials)}, ran under "
                    f"`{ran_under or 'unrecorded'}`, library now `{current_library}`"
                )

    # Action summary at top
    if (
        pending_tasks
        or stage1_done_tasks
        or stage1_running_tasks
        or verify_tasks
        or unusable_reports
    ):
        lines.append("## Next Up")
        lines.append("")
        if unusable_reports:
            lines.append("**Unusable skill reports** (the task's findings cannot be graded):")
            for ref in unusable_reports:
                lines.append(f"- {ref}")
            lines.append("")
        if verify_tasks:
            lines.append("**Skill verification needed** (dispatch a verifier subagent):")
            for ref in verify_tasks:
                lines.append(f"- {ref}")
            lines.append("")
        if stage1_done_tasks:
            lines.append(
                "**Stage 2 needed** (fix_code.py exists, run validation on seeds "
                f"1–{DONE_THRESHOLD}):"
            )
            for ref in stage1_done_tasks:
                lines.append(f"- {ref}")
            lines.append("")
        if stage1_running_tasks:
            lines.append(
                "**Stage 1 incomplete** (fix_code.py written, no skill report yet) — the worker"
                " is still running, or it died after Step 3. Disk cannot tell these apart:"
                " check your ledger. Do NOT start Stage 2 on these; a running worker still owns"
                " the GPU."
            )
            for ref in stage1_running_tasks:
                lines.append(f"- {ref}")
            lines.append("")
        if pending_tasks:
            lines.append("**Pending** (ready for Stage 1):")
            for ref in pending_tasks:
                lines.append(f"- {ref}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Per-(benchmark, suite) tables
    for (benchmark, suite), rows in group_rows.items():
        done = sum(1 for _, _, status, _, _ in rows if status == "done")
        lines.append(f"## {benchmark} / {suite}  ({done}/{len(rows)} done)")
        lines.append("")
        lines.append("| Task | Status | Rate | Skills |")
        lines.append("|---|---|---|---|")
        for task_id, label, status, rate, skill_cell in sorted(rows):
            lines.append(f"| {task_id:02d} {label} | {status} | {rate} | {skill_cell} |")
        lines.append("")

    # The rerun sweep is phase 3: it runs once, after every task is done, so
    # each reruns against the richest library the campaign produced.
    all_done = not pending_tasks and not stage1_done_tasks and not stage1_running_tasks
    if rerun_candidates:
        lines.append("## Rerun sweep")
        lines.append("")
        if all_done:
            lines.append(
                f"Every task is done. These scored below {args.rerun_threshold:.0%} under a "
                "library that has since changed — rerun them against the current one:"
            )
        else:
            lines.append(
                f"These scored below {args.rerun_threshold:.0%} under a library that has since "
                "changed. **Do not rerun yet** — the sweep runs once, after every task is done, "
                "so each rerun sees the final library:"
            )
        lines.append("")
        for candidate in rerun_candidates:
            lines.append(f"- {candidate}")
        lines.append("")

    output = "\n".join(lines)

    if args.print_only:
        print(output)
    else:
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(output)
        tmp.rename(out_file)  # atomic on Linux — safe under concurrent subagent writes
        print(f"Written: {out_file}")

    total_done = sum(
        1 for rows in group_rows.values() for _, _, status, _, _ in rows if status == "done"
    )
    total_tasks = sum(len(rows) for rows in group_rows.values())
    print(
        f"Progress: {total_done}/{total_tasks} tasks done  |  "
        f"{len(stage1_done_tasks)} need Stage 2  |  "
        f"{len(stage1_running_tasks)} mid-Stage 1  |  "
        f"{len(pending_tasks)} pending  |  "
        f"{len(verify_tasks)} need skill verification  |  "
        f"{len(rerun_candidates)} rerun candidate(s)"
    )


if __name__ == "__main__":
    main()
