"""Shared campaign-root helpers for the iterative-debugging pipeline scripts.

A campaign is one pipeline invocation. All of its artifacts live under a single
timestamped root:

    outputs/runs/<stamp>/                      ← campaign root (RUN_ROOT)
      tasks.json                               ← campaign task list
      progress.md                              ← generated status
      <benchmark>/<suite>/task_<id>/           ← one folder per task
        task_analysis.md  findings.md  fix_code.py
        code/                                  ← every program version + CHANGELOG.md
        debug/{explore,initial,attempts,logs,blocked}/   ← Stage 0/1 artifacts
        validation/runs/<run_id>/              ← Stage 2 held-out eval (seeds 1–50)

``outputs/runs/LATEST`` is a symlink to the most recently initialized campaign;
scripts default to it so subagents and the coordinator agree on the root.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

# Repository root (this file lives in .claude/skills/iterative-debugging/scripts/).
ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = ROOT / "outputs/runs"
LATEST_LINK = RUNS_ROOT / "LATEST"
SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = SKILL_ROOT / "skills"


def resolve_run_root(run_root: Path | None) -> Path:
    """Resolve the campaign root: an explicit path, else outputs/runs/LATEST."""
    if run_root is not None:
        resolved = run_root if run_root.is_absolute() else ROOT / run_root
        if not resolved.is_dir():
            raise SystemExit(f"run root does not exist: {resolved}")
        return resolved.resolve()
    if not LATEST_LINK.exists():
        raise SystemExit(
            f"no campaign found: {LATEST_LINK} is missing.\n"
            "Initialize one with scripts/init_run.py, or pass --run-root explicitly."
        )
    return LATEST_LINK.resolve()


def task_dir(run_root: Path, benchmark: str, suite: str, task_id: int) -> Path:
    return run_root / benchmark / suite / f"task_{task_id}"


DEFAULT_HELDOUT_COUNT = 50
DEV_SEEDS = list(range(51, 66))
UNSEEN_SEEDS = list(range(66, 81))


def campaign_settings(run_root: Path | None) -> dict:
    """Per-campaign settings written by init_run.py (``<run_root>/campaign.json``).

    The held-out partition used to be fixed at seeds 1-50. Simulators whose
    episodes cost minutes validate on 1-20 instead, so
    the count is a campaign setting; the default keeps the historical 50.
    Development, unseen-check and held-out are three disjoint seed bands
    (init_run.py enforces this); Robosuite uses its own ranges, e.g.
    development 101-125, unseen-check 126-130, held-out 1-100.
    """
    settings = {
        "heldout_count": DEFAULT_HELDOUT_COUNT,
        "dev_seeds": DEV_SEEDS,
        "unseen_seeds": UNSEEN_SEEDS,
    }
    if run_root is None:
        return settings
    path = Path(run_root) / "campaign.json"
    if path.is_file():
        try:
            settings.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return settings


def heldout_seeds(run_root: Path | None) -> list[int]:
    return list(range(1, int(campaign_settings(run_root)["heldout_count"]) + 1))


def format_seed_range(seeds: list[int]) -> str:
    """Compact display form: a contiguous run renders as ``start-end``, else comma-separated."""
    ordered = sorted(seeds)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}" if len(ordered) > 1 else str(ordered[0])
    return ",".join(str(s) for s in ordered)


def verification_dir(
    run_root: Path, benchmark: str, suite: str, task_id: int, attempt: str | None = None
) -> Path:
    """Where a skill verifier works — deliberately *outside* the task directory.

    The verifier's result only means something if it never saw the solution it
    is meant to rediscover. Sitting its workspace under the task directory put
    ``fix_code.py`` one ``..`` away and, worse, made the template hand over the
    path to everything it was forbidden to read. Here it is told one directory
    and never learns where the proposer's artifacts live.
    """
    name = f"{benchmark}__{suite}__task_{task_id}"
    if attempt:
        name = f"{name}__{attempt}"
    return run_root / "verification" / name


def skill_library_sha(skills_dir: Path | None = None) -> str:
    """Content hash of the shared skill library.

    Recorded on every task at dispatch, so "which library did this task run
    under" is a fact on disk rather than a memory. Names are hashed alongside
    contents, so a renamed or deleted skill changes the hash even when the
    surviving text does not; the digest is over sorted paths, so it does not
    depend on directory iteration order.
    """
    directory = skills_dir if skills_dir is not None else SKILLS_DIR
    digest = hashlib.sha256()
    if not directory.is_dir():
        return digest.hexdigest()[:16]
    # README.md documents the lifecycle rather than teaching the robot
    # anything, so a docs-only edit must not change the library's identity and
    # make every finished task look like it ran under an older one.
    for path in sorted(p for p in directory.rglob("*.md") if p.name != "README.md"):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _main() -> None:
    """`python3 scripts/campaign.py --skill-library-sha` — the hash, for prompts and shells."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-library-sha",
        action="store_true",
        help="print the content hash of skills/ and exit",
    )
    parser.add_argument(
        "--validate-skill-report",
        type=Path,
        metavar="PATH",
        help="check a skill_report.json and exit nonzero if it is unusable",
    )
    args = parser.parse_args()
    if args.skill_library_sha:
        print(skill_library_sha())
        return
    if args.validate_skill_report is not None:
        path = args.validate_skill_report
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")
        try:
            report = json.loads(path.read_text())
        except ValueError as exc:
            raise SystemExit(f"{path}: not valid JSON -- {exc}") from exc
        problems = validate_skill_report(report)
        for problem in problems:
            print(f"  {problem}")
        if problems:
            raise SystemExit(f"{path}: {len(problems)} problem(s); fix before returning")
        print(f"{path}: OK")
        return
    parser.error("nothing to do; pass --skill-library-sha or --validate-skill-report")


SKILL_VERDICTS = ("useful", "misleading", "wrong", "not-applicable")
#: Exception names a generated program may raise. Mirrors ``_SAFE_BUILTINS`` in
#: ``cap_harness.runtime``; these scripts run under a bare ``python3`` that
#: usually cannot import the package, so the list is duplicated here and a test
#: in the harness suite fails if the two ever drift apart. A snippet raising
#: anything else dies with ``NameError`` in every program that copies it --
#: which is exactly how six library snippets shipped `raise RuntimeError`.
PROGRAM_EXCEPTIONS = ("Exception", "ValueError", "TypeError")
#: Below this, a `misleading` or `wrong` grade is an opinion rather than a
#: repair order. "grasp.md was confusing" is 24 characters.
_SPECIFIC_ENOUGH = 40

#: Builtins a generated program may use. Mirrors ``_SAFE_BUILTINS`` in
#: ``cap_harness.runtime`` for the same reason ``PROGRAM_EXCEPTIONS`` does, and
#: the same drift test covers it.
PROGRAM_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)
#: The six injected constructors.
PROGRAM_CONSTRUCTORS = (
    "ArmCommand",
    "MotionStrategy",
    "Pose",
    "RobotAction",
    "SynchronizedTrajectory",
    "Trajectory",
)
#: Public tools a program may call (``docs/api-reference.md``).
PUBLIC_TOOLS = (
    "get_task_context",
    "get_observation",
    "get_robot_state",
    "segment_text",
    "segment_points",
    "mask_to_point_cloud",
    "estimate_geometry",
    "localize_object",
    "generate_grasps",
    "select_grasp",
    "solve_ik",
    "plan_motion",
    "execute_trajectory",
    "move_to_joints",
    "move_to_pose",
    "go_home",
    "step",
    "open_gripper",
    "close_gripper",
    "set_gripper",
    "libero",
    "robosuite",
    "robocasa",
)
#: Names the library itself establishes, so a snippet may use them bare: the two
#: pure-Python helpers it ships and the two conventional bindings every snippet
#: opens with. Everything else a snippet reads without binding is a placeholder
#: the reader has to guess at.
LIBRARY_IDIOMS = ("quantile", "quaternion_product", "base_frame", "downward", "result")
#: Calls that consume SAM3 output, and therefore depend on a prompt string.
PERCEPTION_TOOLS = ("segment_text", "segment_points", "localize_object")
#: Strings that appear beside prompts without being one -- camera names, frames,
#: strategy and backend keys. A snippet whose only literal is ``"agentview"`` has
#: still not said what it segments for.
NON_PROMPT_STRINGS = frozenset(
    {
        "agentview",
        "frontview",
        "birdview",
        "sideview",
        "robot0_eye_in_hand",
        "primary",
        "secondary",
        "base_frame",
        "world",
        "robot_base",
        "top_down",
        "contact-graspnet",
        "graspgen",
        "curobo",
        "pyroki",
    }
)

_SUPPLIED_NAMES = frozenset(
    PROGRAM_BUILTINS + PROGRAM_CONSTRUCTORS + PUBLIC_TOOLS + PROGRAM_EXCEPTIONS + LIBRARY_IDIOMS
)


def snippet_free_names(code: str) -> tuple[frozenset[str], str | None]:
    """Undefined *scalars* a snippet computes with, and a parse error if any.

    A proposal is judged on its text alone, so every name it reads without
    binding is something the reader must invent. Most of those are harmless: a
    snippet is a fragment, and ``knob_points`` or ``part_candidate_clouds`` --
    names that are only iterated over -- say what they are. The ones that cost
    an attempt are the *derived scalars*, where the derivation is the content
    and several readings look equally right. ``fixture_z`` shipped undefined;
    the obvious reading (``max(z)``) landed on a handful of edge-noise points,
    the footprint quantiles degenerated, and the height window then rejected the
    very part the entry existed to select. So the test is not "is this name
    defined" but "does this snippet do arithmetic with a number it never
    derived".

    Binding is over-approximated (a name stored anywhere counts as bound
    everywhere), which errs toward accepting.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError as exc:
        return frozenset(), f"does not parse as Python ({exc.msg} at line {exc.lineno})"
    bound: set[str] = set()
    used: set[str] = set()
    computed: set[str] = set()
    # ``found.ok`` reads an attribute off an object; the object's own numeric
    # derivation is not what the expression depends on, so a name used only that
    # way is not a scalar the reader has to derive.
    attribute_bases = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, ast.Store | ast.Del) else used).add(node.id)
        elif isinstance(node, ast.FunctionDef):
            bound.add(node.name)
            arguments = node.args
            for arg in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ):
                if arg is not None:
                    bound.add(arg.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.BinOp | ast.Compare) or (
            isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not)
        ):
            computed.update(
                inner.id
                for inner in ast.walk(node)
                if isinstance(inner, ast.Name)
                and isinstance(inner.ctx, ast.Load)
                and id(inner) not in attribute_bases
            )
    return frozenset((used & computed) - bound - _SUPPLIED_NAMES), None


def _snippet_calls(code: str, names: tuple[str, ...]) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names
        for node in ast.walk(tree)
    )


def _snippet_string_literals(code: str) -> list[str]:
    """String constants in a snippet, excluding docstrings.

    Docstrings are excluded because a helper's docstring is not a prompt, and
    counting it would let a proposal satisfy the prompt requirement by
    explaining itself.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def validate_skill_report(
    report: object, skills_dir: Path | None = None, *, check_skill_files: bool = True
) -> list[str]:
    """Problems with a ``skill_report.json``, most structural first.

    An empty list means the report is usable. This exists because the failure
    it catches is silent: a malformed report degrades to "this task proposed
    nothing", which reads exactly like a task that had nothing to say.
    """
    if not isinstance(report, dict):
        return ["not a JSON object"]

    problems: list[str] = []
    known = (
        {path.name for path in (skills_dir or SKILLS_DIR).glob("*.md")}
        if check_skill_files and (skills_dir or SKILLS_DIR).is_dir()
        else set()
    )
    if not str(report.get("skill_library_sha") or "").strip():
        problems.append("skill_library_sha is missing — a rerun cannot tell which library ran")

    def check_skill(where: str, name: object) -> None:
        if not str(name or "").strip():
            problems.append(f"{where}: no skill named")
        elif known and str(name) not in known:
            problems.append(f"{where}: no such skill file {name!r} (have {sorted(known)})")

    for index, entry in enumerate(report.get("consulted") or []):
        where = f"consulted[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue
        check_skill(where, entry.get("skill"))
        verdict = entry.get("verdict")
        if verdict not in SKILL_VERDICTS:
            problems.append(f"{where}: verdict {verdict!r} is not one of {list(SKILL_VERDICTS)}")
        evidence = str(entry.get("evidence") or "").strip()
        if not evidence:
            problems.append(f"{where}: no evidence for a {verdict!r} grade")
        elif verdict in ("misleading", "wrong") and len(evidence) < _SPECIFIC_ENOUGH:
            problems.append(
                f"{where}: a {verdict!r} grade needs specifics a maintainer can act on, "
                f"got {evidence!r}"
            )

    for index, entry in enumerate(report.get("proposed_new") or []):
        where = f"proposed_new[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue
        check_skill(where, entry.get("skill"))
        for field in ("title", "trigger", "code", "evidence"):
            if not str(entry.get(field) or "").strip():
                problems.append(f"{where}: {field} is empty — a verifier gets only this text")
        code = str(entry.get("code") or "")
        problems.extend(
            f"{where}: code raises {name}, which programs cannot use "
            f"(allowed: {list(PROGRAM_EXCEPTIONS)})"
            for name in sorted(set(re.findall(r"raise\s+(\w+)\s*\(", code)))
            if name not in PROGRAM_EXCEPTIONS
        )
        if not code.strip():
            continue

        # A verifier is handed this text and nothing else, so a name the snippet
        # reads without binding has to be explained somewhere or it is a guess.
        # Naming it in the trigger or evidence is enough -- the bar is that the
        # author said what it is, not that the snippet is runnable standalone.
        prose = " ".join(str(entry.get(field) or "") for field in ("title", "trigger", "evidence"))
        free, parse_error = snippet_free_names(code)
        if parse_error:
            problems.append(f"{where}: code {parse_error}; a verifier cannot run it")
        unexplained = sorted(
            name for name in free if not re.search(rf"\b{re.escape(name)}\b", prose)
        )
        if unexplained:
            problems.append(
                f"{where}: code uses {unexplained} without defining them or naming them in "
                "the trigger/evidence — the verifier has to guess the derivation, which is "
                "where a proposal loses an attempt"
            )

        # An entry that consumes segmentation but names no prompt hands over the
        # selection logic and withholds the input it runs on. That is how the
        # library's prompt registry stayed empty while the entry that exists
        # *because* a prompt is unreliable said nothing about which prompt.
        needs_prompt = entry.get("skill") == "localize.md" or _snippet_calls(code, PERCEPTION_TOOLS)
        # The prompt has to be in the *code*, not the prose. Prose quotes are
        # not evidence: the entry this check comes from quoted an English phrase
        # ("the knob candidate nearest the fixture centre") and named no prompt
        # at all, which is how a verifier ended up measuring prompts itself.
        if needs_prompt and not (set(_snippet_string_literals(code)) - NON_PROMPT_STRINGS):
            problems.append(
                f"{where}: depends on segmentation but names no prompt in the code — give "
                "the concrete prompt string(s) a reader would pass to segment_text/"
                "localize_object"
            )

    for index, entry in enumerate(report.get("proposed_edits") or []):
        where = f"proposed_edits[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not an object")
            continue
        check_skill(where, entry.get("skill"))
        for field in ("change", "why"):
            if not str(entry.get(field) or "").strip():
                problems.append(f"{where}: {field} is empty")

    return problems


def read_json(path: Path) -> dict | None:
    """Read a JSON artifact, or None if it is missing or unreadable.

    Progress is derived from disk under concurrent writers, so a half-written
    or absent sidecar is an expected state, not an error to raise on.
    """
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


if __name__ == "__main__":
    _main()
