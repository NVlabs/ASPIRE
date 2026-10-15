"""The pipeline prompts are the program. Their commands must still parse.

Subagents, verifiers, and the coordinator run the shell blocks in these
markdown files verbatim. Nothing else checks them: rename a flag in `cli.py` or
`run_validation.py` and every prompt keeps its confident instructions, the test
suite stays green, and the pipeline breaks at the first dispatch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shlex

import pytest

from cap_harness.cli import build_parser

SKILL_ROOT = Path(__file__).resolve().parents[1] / ".claude/skills/iterative-debugging"
PROMPTS = sorted(SKILL_ROOT.glob("*.md")) + sorted(SKILL_ROOT.glob("skills/*.md"))
#: Stand-ins for the shell expansions a prompt leaves for its agent to fill.
PLACEHOLDERS = {
    "$(seq 1 $HELDOUT_COUNT)": " ".join(str(seed) for seed in range(1, 51)),
    "$CAP_HARNESS": ".venv-libero/bin/cap-harness",
    "$INIT_MODE": "seeded",
    "$VALIDATION_WORKERS": "4",
    "$RUN_FLAGS": "--camera-width 512 --camera-height 512",
    "$BENCHMARK": "libero-pro",
    "$SUITE": "libero_goal_swap",
    "$TASK_ID": "0",
    "$GPU": "3",
    "$RUN_ROOT": "/tmp/run",
    "$TASK_DIR": "/tmp/task",
    "$VERIFY_DIR": "/tmp/verify",
    "$trial": "51",
    "$W": "4",
    "<N>": "51",
    "$(seq 1 50)": " ".join(str(seed) for seed in range(1, 51)),
}


def prose(path: Path) -> str:
    """Read a prompt with its hard wrapping and blockquote markers collapsed.

    These files wrap at 100 columns, so a sentence a test cares about is
    routinely split across lines. Matching the raw text fails on rewrapping
    rather than on meaning.
    """
    lines = [line.lstrip().removeprefix(">").strip() for line in path.read_text().splitlines()]
    return " ".join(" ".join(lines).split())


def shell_commands(text: str, executable: str) -> list[list[str]]:
    """Every invocation of ``executable`` in the fenced bash blocks, tokenized."""
    found: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        # Rejoin backslash continuations into single logical lines.
        joined = re.sub(r"\\\n\s*", " ", block)
        for raw in joined.splitlines():
            if executable not in raw and "$CAP_HARNESS" not in raw:
                continue
            expanded = raw
            for placeholder, value in PLACEHOLDERS.items():
                expanded = expanded.replace(placeholder, value)
            # Drop redirections and anything downstream of them.
            expanded = re.split(r"[>|]", expanded)[0]
            try:
                tokens = shlex.split(expanded)
            except ValueError:  # unbalanced quotes in prose
                continue
            # Strip leading VAR=value environment prefixes and the binary path.
            while tokens and re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", tokens[0]):
                tokens.pop(0)
            if not tokens or executable not in tokens[0]:
                continue
            found.append(tokens[1:])
    return found


def test_the_prompts_actually_contain_commands() -> None:
    """Verify that command extraction finds actual prompt commands.

    Guard the guard: if extraction silently finds nothing, every test below
    passes vacuously and the check is worthless.
    """
    total = sum(len(shell_commands(path.read_text(), "cap-harness")) for path in PROMPTS)
    assert total >= 3, f"expected cap-harness invocations across {len(PROMPTS)} prompts"


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda path: path.name)
def test_every_cap_harness_command_in_the_prompts_parses(prompt: Path) -> None:
    parser = build_parser()
    for argv in shell_commands(prompt.read_text(), "cap-harness"):
        try:
            parser.parse_args(argv)
        except SystemExit as exit_error:  # argparse exits on an unknown flag
            raise AssertionError(
                f"{prompt.name}: `cap-harness {' '.join(argv)}` no longer parses"
            ) from exit_error


def _run_validation_flags() -> set[str]:
    path = SKILL_ROOT / "scripts/run_validation.py"
    spec = importlib.util.spec_from_file_location("run_validation_for_flags", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # parse_args() builds and parses in one call, so read the flags from source.
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', path.read_text()))


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda path: path.name)
def test_every_run_validation_flag_in_the_prompts_exists(prompt: Path) -> None:
    known = _run_validation_flags()
    for argv in shell_commands(prompt.read_text(), "run_validation.py"):
        used = {token for token in argv if token.startswith("--")}
        unknown = used - known
        assert not unknown, f"{prompt.name}: run_validation.py has no {sorted(unknown)}"


def test_the_documented_stage_commands_are_the_batched_ones() -> None:
    """Keep stage commands batched to fit the command timeout.

    The loops these replaced could not fit under the 10-minute command
    ceiling, so a silent revert to `cap-harness run` per seed reintroduces
    sweeps that get killed partway.
    """
    subagent = (SKILL_ROOT / "subagent-prompt.md").read_text()
    verifier = (SKILL_ROOT / "verifier-prompt.md").read_text()
    coordinator = (SKILL_ROOT / "main-agent-prompt.md").read_text()

    assert "run-batch" in subagent and "for trial in $(seq 51 65)" not in subagent
    assert "run-batch" in verifier
    assert "--workers" in coordinator


def test_every_debug_version_is_scored_on_the_whole_development_set() -> None:
    """Score every debug version on the full development set.

    The old loop tested each fix on the one seed that motivated it, so a fix
    that repaired seed 53 and broke 51, 52, and 54 looked like progress and the
    regression only surfaced fifty held-out runs later.
    """
    subagent = (SKILL_ROOT / "subagent-prompt.md").read_text()
    sweeps = [argv for argv in shell_commands(subagent, "cap-harness") if "run-batch" in str(argv)]
    assert sweeps, "the debug loop must sweep, not run single seeds"

    seed_args = []
    for argv in sweeps:
        if "--seeds" in argv:
            start = argv.index("--seeds") + 1
            seed_args.append(next(t for t in argv[start:] if not t.startswith("--")))
    # Every debug sweep covers the full development band (the coordinator's seed
    # assignment from campaign.json, e.g. 51-65 for LIBERO); nothing runs one seed.
    assert "$DEV_SEEDS" in seed_args
    assert not any(argv[:2] == ["run", "--seed"] for argv in sweeps)


def test_the_unseen_band_is_measured_once_and_never_tuned_against() -> None:
    """Measure the unseen band once without tuning against it.

    The unseen-check seeds (campaign.json's `unseen_seeds`, e.g. 66-80 for LIBERO,
    126-130 for Robosuite) are the only cheap evidence that a program is not tuned
    to the seeds it was debugged on. Debugging against them spends exactly the
    thing they exist to provide.
    """
    subagent = prose(SKILL_ROOT / "subagent-prompt.md")

    assert "$UNSEEN_SEEDS" in subagent
    assert "do not run them twice" in subagent.lower()
    # The band must be excluded from the debugging scope, not merely mentioned.
    assert "Never debug against" in subagent
    # And it must never overlap the coordinator's held-out partition.
    assert "Do NOT run held-out seeds 1–$HELDOUT_COUNT" in subagent


def test_debugging_is_organised_by_failure_mode() -> None:
    subagent = prose(SKILL_ROOT / "subagent-prompt.md")

    assert "failure mode, not by seed" in subagent
    assert "Budget: 3 versions per mode" in subagent
    # The per-seed attempt budget is what pushed toward seed-specific fixes.
    assert "3 fix attempts per seed" not in subagent


def test_the_verifier_is_barred_from_the_shipped_example_solutions() -> None:
    """Keep shipped example solutions off-limits to the verifier.

    `docs/libero-pro.md` is on the verifier's permitted list and names a
    worked solution per suite and task. A real verifier found that pointer and
    flagged it: reading it would void the measurement as surely as reading the
    proposer's program, and nothing in the rules said so.
    """
    verifier = prose(SKILL_ROOT / "verifier-prompt.md")

    assert "examples/" in verifier
    assert "docs/libero-pro.md" in verifier
    # The bar has to sit in the MUST NOT list, not merely be mentioned.
    forbidden = verifier.split("MUST NOT")[1].split("Budget")[0]
    assert "examples/" in forbidden


def test_the_coordinator_does_not_stack_a_verifier_onto_a_busy_gpu() -> None:
    """Never dispatch a verifier onto an already busy GPU.

    The dispatch instruction used to say 'on the GPU that task already owns,
    alongside its Stage 2 eval', contradicting 'one job per GPU, ever'.
    """
    coordinator = prose(SKILL_ROOT / "main-agent-prompt.md")

    assert "on a free GPU" in coordinator
    assert "alongside its Stage 2 eval" not in coordinator
    assert "one job per gpu" in coordinator.lower()
