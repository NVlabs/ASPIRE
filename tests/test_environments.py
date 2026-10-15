from __future__ import annotations

import json
from pathlib import Path

import pytest

from cap_harness import environments

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("rtx5090", "l40")


def test_topology_loads_and_is_internally_consistent() -> None:
    topo = environments.load_topology(REPO_ROOT)
    assert topo["schema"] == "cap-harness/environments/1"

    # Every venv basename follows the canonical `.venv-` prefix.
    prefix = topo["venv_prefix"]
    for spec in {**topo["simulators"], **topo["providers"]}.values():
        assert spec["venv"].startswith(prefix)

    # Provider ports are unique and cover the required acceptance set.
    ports = [p["port"] for p in topo["providers"].values()]
    assert len(ports) == len(set(ports)), "provider ports must be unique"
    assert set(topo["required_provider_ports"]).issubset(set(ports))

    # Client dependencies are model-server-free (no torch / model packages).
    banned = {"torch", "torchvision", "sam3", "open3d", "curobo", "pyroki"}
    assert not (set(topo["client_dependencies"]) & banned)


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profile_binds_topology_to_hardware(profile_name: str) -> None:
    profile = environments.load_profile(profile_name, REPO_ROOT)
    topo = environments.load_topology(REPO_ROOT)
    known_providers = set(topo["providers"])

    # Architecture must resolve against the gpu_runtime build matrix.
    arch = profile["arch"]
    assert environments.cuda_arch_list(arch, REPO_ROOT)  # raises if unknown

    # GPU placement and concurrency only reference known providers.
    assert set(profile["gpus"]).issubset(known_providers)
    assert set(profile["concurrency"]).issubset(known_providers)

    # Every GPU provider has a placement and a bounded (>=1) concurrency.
    for name, spec in topo["providers"].items():
        resolved = environments.resolve_provider(name, profile, REPO_ROOT, topo)
        assert resolved.port == spec["port"] or resolved.port in profile.get("ports", {}).values()
        assert resolved.concurrency >= 1
        if spec.get("gpu"):
            assert resolved.gpu_index is not None
            assert resolved.cuda_arch_list == environments.cuda_arch_list(arch, REPO_ROOT)


def test_resolve_provider_rejects_unknown() -> None:
    profile = environments.load_profile("rtx5090", REPO_ROOT)
    with pytest.raises(ValueError):
        environments.resolve_provider("does-not-exist", profile, REPO_ROOT)


def test_venv_backing_root_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = {"venv_root": "/mnt/shared/venvs"}
    monkeypatch.delenv("CAP_HARNESS_VENV_ROOT", raising=False)
    assert environments.venv_backing_root(profile) == "/mnt/shared/venvs"
    assert environments.venv_backing_root(None) is None
    monkeypatch.setenv("CAP_HARNESS_VENV_ROOT", "/scratch/node-local")
    assert environments.venv_backing_root(profile) == "/scratch/node-local"


def test_profiles_are_valid_json_with_matching_name() -> None:
    for name in PROFILES:
        data = json.loads((REPO_ROOT / "configs/profiles" / f"{name}.json").read_text())
        assert data["name"] == name
        assert data["schema"] == "cap-harness/profile/1"


def test_provider_extras_are_split_with_aggregate_alias() -> None:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    for name in ("providers-sam3", "providers-pyroki", "providers-graspnet", "providers"):
        assert name in extras, f"missing extra {name}"

    def dist_names(items: list[str]) -> set[str]:
        out = set()
        for item in items:
            token = item.split(";")[0].strip()
            for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
                token = token.split(sep)[0]
            out.add(token.strip().lower())
        return out

    # pyroki server carries no torch/model stack (it is jax-based).
    assert "torch" not in dist_names(extras["providers-pyroki"])
    # The aggregate remains a true union alias of the per-provider extras.
    union = (
        dist_names(extras["providers-sam3"])
        | dist_names(extras["providers-pyroki"])
        | dist_names(extras["providers-graspnet"])
    )
    assert dist_names(extras["providers"]) == union

    # Simulator extras never pull model-server packages (client-only).
    model_server = {"sam3", "open3d", "contact-graspnet-pytorch", "pyroki", "trimesh", "pyrender"}
    for sim in ("robosuite", "libero"):
        assert not (dist_names(extras[sim]) & model_server), sim


def test_provider_clients_are_import_safe_without_model_stacks() -> None:
    import ast

    client_dir = REPO_ROOT / "src/cap_harness/providers"
    banned = {"torch", "torchvision", "open3d", "sam3", "trimesh", "pyrender", "jax", "curobo"}
    for client in client_dir.glob("*/client.py"):
        tree = ast.parse(client.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert not (imported & banned), f"{client} imports model stack: {imported & banned}"


def test_bootstraps_use_canonical_repo_local_venv_paths() -> None:
    scripts = REPO_ROOT / "scripts"
    # No bootstrap or validation script defaults a venv to ephemeral /tmp storage.
    for script in list(scripts.glob("bootstrap_*.sh")) + list(scripts.glob("validate_*.sh")):
        text = script.read_text(encoding="utf-8")
        assert "/tmp/${USER:-root}/venv-" not in text, script
    # Every venv-owning bootstrap references the canonical <repo>/.venv-* backing.
    for name in ("libero", "robosuite", "curobo"):
        text = (scripts / f"bootstrap_{name}.sh").read_text(encoding="utf-8")
        assert ".venv-" in text, name
    # Source-build + sim bootstraps route through the shared backing helper.
    for name in ("libero", "robosuite", "curobo", "provider"):
        text = (scripts / f"bootstrap_{name}.sh").read_text(encoding="utf-8")
        assert "venv_backing.sh" in text, name


def test_doctor_service_targets_are_topology_driven() -> None:
    from cap_harness import doctor

    default = doctor.topology_service_targets()
    names = [n for n, _ in default]
    # Default probes the always-on providers (not source-built curobo).
    assert "sam3" in names and "pyroki" in names
    assert "curobo" not in names
    # A full four-provider stack resolves ports from the topology, not hard-coded.
    full = dict(doctor.topology_service_targets(["sam3", "contact_graspnet", "pyroki", "curobo"]))
    topo = environments.load_topology(REPO_ROOT)
    for name, port in full.items():
        assert port == topo["providers"][name]["port"]
    with pytest.raises(ValueError):
        doctor.topology_service_targets(["nope"])


def test_all_bootstraps_record_a_fingerprint_for_noop_rebuilds() -> None:
    scripts = REPO_ROOT / "scripts"
    bootstraps = [
        "bootstrap_provider.sh",
        "bootstrap_libero.sh",
        "bootstrap_robosuite.sh",
        "bootstrap_curobo.sh",
    ]
    for name in bootstraps:
        text = (scripts / name).read_text(encoding="utf-8")
        assert "fingerprint.sh" in text, name
        assert "venv_fingerprint_current" in text, name
        assert "write_venv_fingerprint" in text, name
        # No bootstrap wipes an environment with --clear; unchanged is a no-op.
        assert "uv venv --clear" not in text, name


def test_bimanual_plan_providers_map_to_required_topology_ports() -> None:
    from cap_harness.validation.plan import load_plan

    plan = load_plan(REPO_ROOT / "configs/validation/robosuite-bimanual.yaml")
    profile = environments.load_profile("rtx5090", REPO_ROOT)
    ports = {
        environments.resolve_provider(name, profile, REPO_ROOT).port
        for name in plan.required_providers
    }
    assert ports == set(environments.required_provider_ports(REPO_ROOT))


def test_environment_architecture_doc_is_present_and_linked() -> None:
    doc = REPO_ROOT / "docs/environments.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    for token in ("configs/environments.json", "CAP_HARNESS_VENV_ROOT", "bootstrap_providers.sh"):
        assert token in text
    assert "docs/environments.md" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_profiles_resolve_from_the_gpu_compute_capability() -> None:
    assert environments.arch_for_compute_capability("8.9") == "sm_89"
    assert environments.arch_for_compute_capability("12.0") == "sm_120"
    assert environments.profile_for_gpu("sm_89", "NVIDIA GeForce RTX 4090", REPO_ROOT) == "rtx4090"
    assert environments.profile_for_gpu("sm_89", "NVIDIA L40", REPO_ROOT) == "l40"
    assert environments.profile_for_gpu("sm_89", "", REPO_ROOT) in {"rtx4090", "rtx5880"}
    assert environments.profile_for_gpu("sm_120", "NVIDIA GeForce RTX 5090", REPO_ROOT) == "rtx5090"
    assert environments.profile_for_gpu("sm_10", "", REPO_ROOT) is None
    with pytest.raises(ValueError):
        environments.arch_for_compute_capability("ada")
