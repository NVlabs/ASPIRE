"""Lint the skill library the way the runtime and its readers will actually see it.

Every check here is retroactive: each one names a defect that cost a real run during the 80-task
campaign, and none was invented from first principles. Three are errors; the prose heuristics are
advisory and never fail the build.

Errors:

1. **Sandbox rejection.** Every ```python fence is parsed and pushed through the harness's own
   `_ProgramValidator`. A fence that the validator refuses is an entry that produces
   `program_error` on every seed for anyone who pastes it. One `getattr()` in the most-copied
   helper in the library did exactly that to three agents in a row.

2. **Undefined helpers.** A fence may call the public harness API, a Python builtin, or a
   function some fence in the library defines. Anything else is a `NameError` waiting for the
   next reader. `quantile()` was called 89 times across two files that never defined it, and
   three verifiers lost their first run to it.

3. **A search with no empty case.** A result variable seeded with a real value before a search
   loop, or a `min`/`max` over a name nothing checks for emptiness. Ten refutations, and this
   shape caused three of them: `quat = downward` before a tilt search meant "no tilt matched"
   proceeded with the one orientation the entry's own table ruled out, dropping the object in
   mid-air on 7 of 10 legs.

Advisory (heuristics over prose, never fail the build):

4. **A prescribed loop with no tick cost.** The 1000-tick ceiling decided more verification
   attempts than any geometric error -- including one entry recommending 4-5 strokes where a
   single stroke costs 550-630 ticks.
5. **A reach figure that never says what was in the gripper.** Two tasks published an envelope
   measured empty; carrying the object, the tool froze 7-9 cm short and the lip stripped it.

The undefined-name check also prints cross-file dependencies -- a helper defined in one skill and
called from another. Those are legal but are the ones readers miss, so entries relying on them
should say so explicitly.

Run from `groot/cap`:

    uv run python .claude/skills/iterative-debugging/scripts/lint_skills.py

Exits non-zero on any error-level finding. Changelog files are skipped: they are history, not
runtime guidance.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import pathlib
import re
import sys

SKILLS = pathlib.Path(__file__).resolve().parents[1] / "skills"
API_REFERENCE = pathlib.Path(__file__).resolve().parents[4] / "docs" / "api-reference.md"

FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
SAFE_NAMES = set(dir(builtins))


def public_api() -> set[str]:
    """Callable names the docs advertise, read from the api-reference table.

    Parsed rather than hardcoded so the lint does not drift from the contract it checks against.
    """
    if not API_REFERENCE.exists():
        return set()
    text = API_REFERENCE.read_text()
    # Tools are lower_snake_case; the injected contract objects are CapitalCase
    # constructors (Pose, MotionStrategy, RobotAction, ArmCommand, Trajectory,
    # SynchronizedTrajectory). Both are advertised in the same table, so read
    # both -- hardcoding only `Pose` made every proposal that used any of the
    # others fail as "defined nowhere". That went unnoticed until a campaign
    # required `MotionStrategy`, because no earlier proposal had named one.
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(", text))


def fences(path: pathlib.Path):
    """Yield (line_number, code) for each python fence, with blockquote markers stripped."""
    text = path.read_text()
    for match in FENCE.finditer(text):
        code = "\n".join(
            line[2:] if line.startswith("> ") else (line[1:] if line == ">" else line)
            for line in match.group(1).split("\n")
        )
        yield text[: match.start()].count("\n") + 2, code


def seeded_search(tree: ast.AST) -> list[str]:
    """A search loop whose result was pre-seeded with a real value instead of None.

    This is the exact shape of the most expensive defect the campaign found: `quat = downward`
    before a `for` loop that overwrites `quat` only on success, so "nothing matched" silently
    proceeds with the known-bad default. Cost: 7 of 10 legs, object dropped in mid-air, 0/5.
    Seeding with None forces the empty case to be handled.
    """
    seeded = {
        node.targets[0].id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    }
    found = []
    for loop in [n for n in ast.walk(tree) if isinstance(n, ast.For | ast.While)]:
        if not any(isinstance(n, ast.Break) for n in ast.walk(loop)):
            continue
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in seeded
                and seeded[node.targets[0].id].lineno < loop.lineno
            ):
                found.append(node.targets[0].id)
    return sorted(set(found))


def unguarded_reduction(tree: ast.AST, code: str) -> list[str]:
    """`min(...)`/`max(...)` with a `key=` over a name, with nothing checking it is non-empty.

    Two verifiers lost a run to this: a selector whose candidate list came back empty and the
    reduction raised. An emptiness branch is the fix, and it is also where the fallback goes.
    """
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max"}
            and any(kw.arg == "key" for kw in node.keywords)
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            name = node.args[0].id
            if f"not {name}" not in code and f"len({name})" not in code:
                found.append(f"{node.func.id}({name}, key=...)")
    return sorted(set(found))


PRESCRIBES_LOOP = re.compile(
    r"\b(repeat|loop of|until it|strokes|rungs|increments)\b", re.IGNORECASE
)
# Deliberately narrow. "depth" and "clearance" are everywhere in a vision context, so an earlier
# version of this flagged the Prompt Registry and 13 other innocents -- a warning channel people
# learn to ignore is worse than no channel. Require a reach-ENVELOPE word next to a metre-scale
# number, which is what an empty-gripper measurement actually looks like.
QUOTES_REACH = re.compile(
    r"\b(reach(?:es|ed|able)?|insertion|envelope)\b[^.\n]{0,60}?\b0\.\d{2,4}\b", re.IGNORECASE
)
STATES_LOAD = re.compile(
    r"\b(empty|held|holding|carrying|loaded|in the jaw|gripper's contents|gripped)\b", re.IGNORECASE
)


SECTION_REF = re.compile(r"`?(\w+\.md)`?\s*§\*?([^*\n.,;)]{4,60})")


def dangling_refs(files: dict[str, str]) -> list[str]:
    """Cross-file `other.md` §*Section* pointers that no longer resolve to a heading.

    Entries reference each other constantly, and a reference a reader cannot follow is exactly how
    the campaign's worst failure happened: a snippet was shipped in its pre-correction form because
    the correction lived somewhere the reader never reached. Four parallel rewrites split, merged
    and renamed sections independently and left 15 of these behind, so this is now checked.

    Matching is prefix-based on purpose: headings get re-worded far more often than they get
    genuinely deleted, and a fuzzy match keeps this from crying wolf on a rename that still lands
    somewhere sensible.
    """

    def normalise(text: str) -> str:
        """Strip the markdown a heading and a reference to it spell differently.

        BOTH sides must go through this. The first version normalised only the headings, so a
        reference written with backticks -- §*`planner_failed` Mid-Carry* -- could never match
        anything and was reported dangling forever. A check with a permanent false positive is a
        check people switch off, which is worse than not having written it.
        """
        return re.sub(r"[`*~]", "", text).strip().lower()

    headings = {
        name: {normalise(h) for h in re.findall(r"^#{2,3} (.+)$", text, re.MULTILINE)}
        for name, text in files.items()
    }
    found = []
    for name, text in files.items():
        for match in SECTION_REF.finditer(text):
            target, section = match.group(1), normalise(match.group(2))
            if target not in headings:
                continue
            if not any(section[:30] in h or h[:30] in section for h in headings[target]):
                found.append(f"{name} -> {target} §{match.group(2).strip()[:46]}")
    return sorted(set(found))


def prose_warnings(path: pathlib.Path) -> list[str]:
    """Two heuristics over entry prose, for the causes that are not visible in the AST.

    Both are advisory: they flag an entry for a human to look at, and never fail the build.
    Derived from the refutation post-mortem -- five losses to an unstated tick budget, two to a
    reach figure measured with an empty gripper and applied to a loaded one.
    """
    warnings = []
    text = path.read_text()
    # Split on level-2 headings so each entry is judged on its own.
    entries = re.split(r"\n## ", text)
    for entry in entries[1:]:
        title = entry.split("\n", 1)[0].strip()
        if PRESCRIBES_LOOP.search(entry) and "tick" not in entry.lower():
            warnings.append(
                f"{path.name}: '{title[:54]}' prescribes repetition, quotes no tick cost"
            )
        if QUOTES_REACH.search(entry) and not STATES_LOAD.search(entry):
            warnings.append(
                f"{path.name}: '{title[:54]}' quotes reach/depth, never says what was held"
            )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint the skill library, or one staged proposal.")
    parser.add_argument(
        "--proposal",
        type=pathlib.Path,
        default=None,
        help="Lint a staged proposal (proposed_skills.md) instead of the library. The library is "
        "still parsed, so a helper it already defines counts as defined, but only problems "
        "originating in the proposal are reported. Use this BEFORE dispatching a verifier: "
        "three verifiers in one campaign each lost a run to a NameError this finds instantly.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "src"))
    from cap_harness.runtime import _ProgramValidator

    library = [p for p in sorted(SKILLS.glob("*.md")) if not p.name.endswith("-changelog.md")]
    if args.proposal is not None:
        if not args.proposal.exists():
            raise SystemExit(f"no such proposal: {args.proposal}")
        targets, context = [args.proposal], library
    else:
        targets, context = library, []

    api = public_api()
    rejected: list[str] = []
    empty_case: list[str] = []
    defined: dict[str, set[str]] = {}
    called: dict[str, set[str]] = {}
    warnings: list[str] = []
    sources: dict[str, str] = {}

    for path in context + targets:
        report = path in targets  # context files are read for their definitions only
        if report:
            warnings.extend(prose_warnings(path))
        sources[path.name] = path.read_text()
        for line_no, code in fences(path):
            where = f"{path.name}:{line_no}"
            try:
                tree = ast.parse(code)
            except SyntaxError as error:
                if report:
                    rejected.append(f"{where}  SYNTAX: {error.msg}")
                continue
            if report:
                try:
                    _ProgramValidator().visit(tree)
                except ValueError as error:
                    rejected.append(f"{where}  REJECTED: {error}")
                for name in seeded_search(tree):
                    empty_case.append(
                        f"{where}  '{name}' seeded before a search loop -- seed it None"
                    )
                for call in unguarded_reduction(tree, code):
                    empty_case.append(f"{where}  {call} with no emptiness guard")
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined.setdefault(node.name, set()).add(path.name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and report:
                    called.setdefault(node.func.id, set()).add(where)

    undefined = {
        name: sites
        for name, sites in called.items()
        if name not in SAFE_NAMES and name not in api and name not in defined
    }

    print("=== sandbox validation ===")
    for line in rejected:
        print(f"  {line}")
    print(f"  {len(rejected)} rejection(s)")

    print("\n=== called but defined nowhere ===")
    for name in sorted(undefined):
        print(f"  {name:24s} {sorted(undefined[name])[:4]}")
    print(f"  {len(undefined)} undefined name(s)")

    print("\n=== a search with no empty case ===")
    for line in empty_case:
        print(f"  {line}")
    print(f"  {len(empty_case)} unhandled empty case(s)")

    print("\n=== cross-file helpers (legal, but say so in the entry) ===")
    for name, homes in sorted(defined.items()):
        users = {site.split(":")[0] for site in called.get(name, set())} - homes
        if users:
            print(f"  {name:24s} {sorted(homes)} -> {sorted(users)}")

    broken = [
        line
        for line in dangling_refs(sources)
        if args.proposal is None or any(t.name in line for t in targets)
    ]
    print("\n=== dangling cross-references ===")
    for line in broken:
        print(f"  {line}")
    print(f"  {len(broken)} reference(s) to a heading that no longer exists")

    print("\n=== advisory (heuristic, never fails the build) ===")
    for line in warnings:
        print(f"  {line}")
    print(f"  {len(warnings)} advisory warning(s)")

    return 1 if rejected or undefined or empty_case or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
