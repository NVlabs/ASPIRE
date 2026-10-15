from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from cap_harness.api import CapApi
from cap_harness.registry import (
    BEHAVIOR_PUBLIC_TOOL_NAMES,
    LIBERO_PUBLIC_TOOL_NAMES,
    ROBOSUITE_PUBLIC_TOOL_NAMES,
    SHARED_PUBLIC_TOOL_NAMES,
    ToolRegistry,
)
from cap_harness.runtime import _SAFE_CONSTRUCTORS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_lock_has_exact_required_git_pins() -> None:
    lock = json.loads(
        (REPOSITORY_ROOT / "configs/dependency-lock.json").read_text(encoding="utf-8")
    )
    expected_dependencies = {
        "behavior_1k",
        "contact_graspnet_pytorch",
        "libero_pro",
        "pyroki",
        "robosuite",
        "sam3",
    }
    # Every pinned dependency comes from one of these repositories and no other.
    # An exact allowlist, rather than a substring denylist, so that a new remote
    # fails this test by default and renaming this repository cannot affect it.
    approved_repositories = {
        "github.com/uynitsuj/LIBERO-PRO",
        "github.com/Max-Fu/robosuite",
        "github.com/Max-Fu/sam3",
        "github.com/uynitsuj/contact_graspnet_pytorch",
        "github.com/chungmin99/pyroki",
        "github.com/StanfordVL/BEHAVIOR-1K",
    }

    assert lock["schema_version"] == 2
    assert set(lock["dependencies"]) == expected_dependencies
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", dependency["commit"])
        for dependency in lock["dependencies"].values()
    )
    repositories = {
        re.sub(r"^https://|\.git$", "", dependency["repository"])
        for dependency in lock["dependencies"].values()
    }
    assert repositories == approved_repositories
    # gpu_runtime is a generic, hardware-neutral architecture contract: shared
    # torch/torchvision/cuda versions plus per-architecture build targets. It
    # replaces the misleadingly named l40_runtime key.
    assert "l40_runtime" not in lock
    gpu_runtime = lock["gpu_runtime"]
    assert gpu_runtime["cuda_runtime"] == "12.8"
    assert gpu_runtime["platform"] == "linux-x86_64"
    assert gpu_runtime["torch"] == "2.9.1"
    assert gpu_runtime["torchvision"] == "0.24.1"
    assert gpu_runtime["torch_index"] == "cu128"
    # Blackwell (sm_120) and Ada/L40 (sm_89) build targets are both preserved.
    assert gpu_runtime["arches"]["sm_120"]["cuda_arch_list"] == "12.0"
    assert gpu_runtime["arches"]["sm_89"]["cuda_arch_list"] == "8.9"
    assert "rtx5090" in gpu_runtime["arches"]["sm_120"]["profiles"]
    assert "l40" in gpu_runtime["arches"]["sm_89"]["profiles"]
    assert set(lock["optional_dependencies"]) == {"curobo"}
    assert lock["optional_dependencies"]["curobo"]["license"] == "Apache-2.0"
    # Contact-GraspNet ships its weights under a non-commercial licence; the lock
    # says so, so nothing can pretend otherwise.
    assert "non-commercial" in lock["dependencies"]["contact_graspnet_pytorch"]["license"]


def test_standalone_runtime_owns_direct_submodules_and_service_entrypoints() -> None:
    gitmodules = (REPOSITORY_ROOT / ".gitmodules").read_text(encoding="utf-8")
    for path in (
        "third_party/LIBERO-PRO",
        "third_party/contact_graspnet_pytorch",
        "third_party/robosuite",
        "third_party/sam3",
        "third_party/curobo",
        "third_party/BEHAVIOR-1K",
    ):
        assert f"path = {path}" in gitmodules
    assert gitmodules.count("[submodule ") == 6

    bootstrap = (REPOSITORY_ROOT / "scripts/bootstrap_libero.sh").read_text(encoding="utf-8")
    supervisor = (REPOSITORY_ROOT / "scripts/supervise_services.sh").read_text(encoding="utf-8")
    # The bootstraps and the supervisor install and start only what the lock and
    # the topology declare; they never fetch a reference implementation.
    for text in (bootstrap, supervisor):
        assert "git clone" not in text
        assert "pip install http" not in text

    # Service entry points now live in the declarative topology (single source),
    # not hard-coded in the supervisor; the supervisor resolves them from there.
    topology = json.loads(
        (REPOSITORY_ROOT / "configs/environments.json").read_text(encoding="utf-8")
    )
    modules = {spec.get("module") for spec in topology["providers"].values()}
    assert "cap_harness.providers.sam3.service" in modules
    assert "cap_harness.providers.graspnet.service" in modules
    assert "cap_harness.providers.pyroki.service" in modules
    assert "cap_harness.providers.curobo.service" in modules
    assert "from cap_harness import environments" in supervisor
    # Provider services resolve vendored checkpoints via CAP_HARNESS_VENDOR_ROOT;
    # the supervisor must export it (Contact-GraspNet fails to load weights otherwise).
    assert "CAP_HARNESS_VENDOR_ROOT" in supervisor


def test_robosuite_bimanual_runner_records_its_acceptance_contract() -> None:
    runner = (REPOSITORY_ROOT / "scripts/validate_robosuite_bimanual.sh").read_text(
        encoding="utf-8"
    )

    # Thin wrapper: renderer/GL/GPU env + provider preflight, then the sealed plan.
    assert "for port in 8114 8116 8118" in runner
    assert "cap_harness.cli validate --plan" in runner
    assert "configs/validation/robosuite-bimanual.yaml" in runner
    # The case matrix, sealing, and verification live in the plan, not the shell.
    assert "SEEDS=(" not in runner
    assert "cap_harness.robosuite.campaign" not in runner

    plan = (REPOSITORY_ROOT / "configs/validation/robosuite-bimanual.yaml").read_text(
        encoding="utf-8"
    )
    assert "sealed: true" in plan
    assert "evaluator: robosuite_bimanual" in plan
    # Accepted regression floors: Lift 3/3, Handover 1/3.
    assert "min_success: 3" in plan
    assert "min_success: 1" in plan


def test_fixed_bimanual_programs_encode_required_public_api_ordering() -> None:
    lift = (REPOSITORY_ROOT / "examples/robosuite/two_arm_lift_seed1.py").read_text()
    handover = (REPOSITORY_ROOT / "examples/robosuite/two_arm_handover_seed1.py").read_text()

    assert lift.count("set_grippers(") == 1
    assert "move_synchronized(" in lift
    assert 'close_gripper(arm="primary")' not in lift
    assert 'close_gripper(arm="secondary")' not in lift
    assert 'report["lifted"] = lift_ok' in lift
    receiver_close = handover.index('close_gripper(arm="secondary")')
    giver_open = handover.rindex('open_gripper(arm="primary")')
    assert receiver_close < giver_open
    assert 'report["giver_release_commanded"] = giver_release.ok' in handover
    assert 'report["transferred"]' not in handover


def test_fixed_bimanual_programs_are_seed_agnostic_public_api_only() -> None:
    allowed_calls = SHARED_PUBLIC_TOOL_NAMES | {
        "ArmCommand",
        "Pose",
        "RobotAction",
        "float",
        "len",
        "max",
        "range",
        "sorted",
        "sum",
        "tuple",
    }
    for name in ("two_arm_lift_seed1.py", "two_arm_handover_seed1.py"):
        source = (REPOSITORY_ROOT / "examples/robosuite" / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        assert not any(isinstance(node, ast.Import | ast.ImportFrom) for node in ast.walk(tree))
        assert not any(isinstance(node, ast.Name) and node.id == "seed" for node in ast.walk(tree))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert all(isinstance(call.func, ast.Name) for call in calls)
        assert {call.func.id for call in calls} <= allowed_calls
        arm_values = {
            keyword.value.value
            for call in calls
            for keyword in call.keywords
            if keyword.arg == "arm" and isinstance(keyword.value, ast.Constant)
        }
        assert arm_values <= {"primary", "secondary"}


def test_handover_program_uses_public_geometry_without_privileged_state() -> None:
    # The production handover program (not the experimental stage probes, which
    # are deliberately excluded from the clean reconstruction) must obtain its
    # transfer geometry from public perception + crop_point_cloud, keep the
    # giver's grasp until after the receiver closes, and never read native state.
    program = (REPOSITORY_ROOT / "examples/robosuite/two_arm_handover_seed1.py").read_text(
        encoding="utf-8"
    )

    assert 'segment_text("agentview", "wooden hammer handle")' in program
    assert 'segment_text("agentview", "hammer head")' in program
    assert 'close_gripper(arm="primary")' in program
    assert 'close_gripper(arm="secondary")' in program
    assert "crop_point_cloud(" in program
    assert "receiver_quaternion = secondary_pose.quaternion_wxyz" in program
    assert "canonical_quaternion = (0.0, half_cos, half_sin, 0.0)" in program
    assert "transfer_x =" in program
    assert "receiver_plan = plan_motion(receiver_pose" in program
    assert "receiver_plan.trajectory.joint_positions[-1]" in program
    assert "plan_synchronized_motion" not in program
    assert program.rindex('open_gripper(arm="primary")') > program.index(
        'close_gripper(arm="secondary")'
    )
    assert 'report["giver_release_commanded"] = giver_release.ok' in program
    assert "native" not in program


def test_embodiment_guides_are_symmetric_and_linked() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    libero = (REPOSITORY_ROOT / "docs/libero-pro.md").read_text(encoding="utf-8")
    robosuite = (REPOSITORY_ROOT / "docs/robosuite.md").read_text(encoding="utf-8")

    assert all(path in readme for path in ("docs/libero-pro.md", "docs/robosuite.md"))
    for guide in (libero, robosuite):
        assert "## Setup" in guide
        assert "## Semantics" in guide
        assert "## Validation" in guide
        assert "## Recorded run" in guide


def test_runtime_adapter_is_not_exported_from_public_libero_package() -> None:
    from cap_harness import libero

    assert not hasattr(libero, "LiberoAdapter")
    assert not hasattr(libero, "LiberoRuntimeExtensions")


class _ApiDocumentationAdapter:
    embodiment = "libero"

    def get_robot_state(self):
        raise RuntimeError("documentation registry does not need a reset adapter")

    def get_task_metadata(self):
        return {}

    def get_controller_metadata(self):
        return {}


def test_api_reference_tracks_registered_surfaces_and_is_linked() -> None:
    reference_path = REPOSITORY_ROOT / "docs/api-reference.md"
    reference = reference_path.read_text(encoding="utf-8")
    rows = {
        (layer, name)
        for layer, name in re.findall(
            r"^\| `(shared_atomic|shared_high_level)` \| `([a-z_]+)\(",
            reference,
            flags=re.MULTILINE,
        )
    }
    registry = CapApi(_ApiDocumentationAdapter()).register_tools(ToolRegistry())
    registered = {
        (spec.layer, spec.name)
        for spec in registry.all_specs(public_only=True)
        if spec.layer.startswith("shared_")
    }
    assert rows == registered

    documented_extensions = set(
        re.findall(
            r"^\| `extension` \| `([a-z._]+)\(",
            reference,
            flags=re.MULTILINE,
        )
    )
    assert documented_extensions == set().union(
        BEHAVIOR_PUBLIC_TOOL_NAMES,
        LIBERO_PUBLIC_TOOL_NAMES,
        ROBOSUITE_PUBLIC_TOOL_NAMES,
    )

    documented_constructors = set(
        re.findall(r"^\| `([A-Z][A-Za-z]+)` \| `\1\(", reference, flags=re.MULTILINE)
    )
    assert documented_constructors == set(_SAFE_CONSTRUCTORS)

    linked_files = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs/libero-pro.md",
        REPOSITORY_ROOT / "docs/robosuite.md",
    )
    assert all("api-reference.md" in path.read_text(encoding="utf-8") for path in linked_files)
