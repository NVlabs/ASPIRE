#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
VENV_ROOT="$(resolve_canonical_venv "$REPO_ROOT" ".venv-libero")"
UV_CACHE_ROOT="$REPO_ROOT/.uv-cache"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"
INSTALL=1

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_libero.sh [options]

Initialize and install the editable cap-harness checkout in `.venv-libero`.

Options:
  --libero-config-dir PATH    LIBERO configuration directory
                              (default: <repo>/.libero).
  --python PATH               Python interpreter (default: python3).
  --no-install                Initialize dependencies and configuration only.
  -h, --help                  Show this help.

The repository is the single source of truth. This script does not copy source
or start long-running services, and it never reads or writes credentials.
EOF
}

while (($#)); do
  case "$1" in
    --libero-config-dir|--libero-config)
      LIBERO_CONFIG_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-install)
      INSTALL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  printf 'error: Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
}
command -v git >/dev/null 2>&1 || {
  printf 'error: git is required\n' >&2
  exit 1
}
command -v uv >/dev/null 2>&1 || {
  printf 'error: uv is required for the frozen environment\n' >&2
  exit 1
}

DEPENDENCY_LOCK="$REPO_ROOT/configs/dependency-lock.json"
[[ -f "$REPO_ROOT/pyproject.toml" && -f "$REPO_ROOT/uv.lock" ]] || {
  printf 'error: cap-harness project or uv.lock missing under %s\n' "$REPO_ROOT" >&2
  exit 1
}
[[ -f "$DEPENDENCY_LOCK" ]] || {
  printf 'error: dependency lock missing: %s\n' "$DEPENDENCY_LOCK" >&2
  exit 1
}

# The LIBERO runtime needs only these submodules.
SUBMODULES=(
  third_party/LIBERO-PRO
  third_party/contact_graspnet_pytorch
  third_party/robosuite
  third_party/sam3
)

git -C "$REPO_ROOT" submodule update --init "${SUBMODULES[@]}"

"$PYTHON_BIN" - "$DEPENDENCY_LOCK" "$REPO_ROOT" "${SUBMODULES[@]}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

lock_path, checkout = map(Path, sys.argv[1:3])
required = set(sys.argv[3:])
lock = json.loads(lock_path.read_text(encoding="utf-8"))

verified = set()
for name, dependency in lock["dependencies"].items():
    path_value = dependency.get("path")
    if path_value not in required:
        continue
    path = checkout / path_value
    gitlink = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-files", "--stage", "--", path_value],
        text=True,
    ).split()
    expected = dependency["commit"]
    if len(gitlink) < 2 or gitlink[0] != "160000" or gitlink[1] != expected:
        actual = gitlink[1] if len(gitlink) >= 2 else "missing"
        raise SystemExit(f"gitlink mismatch for {path_value}: {actual} != {expected}")
    # An uninitialized submodule is an empty directory, and `rev-parse HEAD` inside it
    # resolves against the parent repository instead of failing.  Require that the path
    # is its own checkout so that case reports itself rather than a bogus mismatch.
    toplevel = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True
    ).strip()
    if Path(toplevel).resolve() != path.resolve():
        raise SystemExit(f"submodule is not initialized: {path_value}")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"checkout mismatch for {name}: {actual} != {expected}")
    verified.add(path_value)

unlocked = required - verified
if unlocked:
    raise SystemExit(f"dependency lock has no entry for: {', '.join(sorted(unlocked))}")
PY

LIBERO_SOURCE="$REPO_ROOT/third_party/LIBERO-PRO"
mkdir -p "$LIBERO_SOURCE/libero/datasets" "$LIBERO_CONFIG_DIR"

"$PYTHON_BIN" - "$LIBERO_CONFIG_DIR/config.yaml" "$LIBERO_SOURCE" <<'PY'
from pathlib import Path
import json
import os
import sys
import tempfile

config_path = Path(sys.argv[1]).expanduser().resolve()
submodule = Path(sys.argv[2]).resolve()
libero_root = submodule / "libero" / "libero"
payload = {
    "benchmark_root": str(libero_root),
    "bddl_files": str(libero_root / "bddl_files"),
    "init_states": str(libero_root / "init_files"),
    "datasets": str(submodule / "libero" / "datasets"),
    "assets": str(libero_root / "assets"),
}
config_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, config_path)
finally:
    try:
        Path(temporary_name).unlink()
    except FileNotFoundError:
        pass
PY

if [[ "$INSTALL" -eq 1 ]]; then
  # Create the environment explicitly, as the sibling bootstrap scripts do.  Without this
  # `uv sync` selects its own interpreter and --python/CAP_HARNESS_PYTHON is silently ignored.
# No-op re-bootstrap when the resolved deps, arch, and pins are unchanged.
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "${CAP_HARNESS_PROFILE:-rtx5090}" "sim:libero:libero" "third_party/LIBERO-PRO" "third_party/robosuite" "third_party/sam3" "third_party/contact_graspnet_pytorch")"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf '%s is up to date (fingerprint %s); skipping rebuild\n' "sim:libero:libero" "${FINGERPRINT:0:12}"
  exit 0
fi
  UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
  # Client-only: install the LIBERO simulator + cap-harness runtime/clients,
  # never the model-serving `providers` stack. Providers run out of their own
  # per-provider venvs and are reached over HTTP.
  VIRTUAL_ENV="$VENV_ROOT" \
    UV_PROJECT_ENVIRONMENT="$VENV_ROOT" \
    UV_CACHE_DIR="$UV_CACHE_ROOT" \
    uv sync --frozen --active \
      --project "$REPO_ROOT" \
      --extra libero \
      --extra dev

  "$VENV_ROOT/bin/python" - "$DEPENDENCY_LOCK" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["gpu_runtime"]
actual = {
    "torch": importlib.metadata.version("torch").split("+", 1)[0],
    "torchvision": importlib.metadata.version("torchvision").split("+", 1)[0],
    "cuda_runtime": torch.version.cuda,
}
# gpu_runtime carries shared versions plus hardware-neutral build metadata
# (platform, torch_index, per-architecture targets); validate only the versions.
for name in ("torch", "torchvision", "cuda_runtime"):
    if actual[name] != runtime[name]:
        raise SystemExit(f"runtime mismatch for {name}: {actual[name]} != {runtime[name]}")
print(
    "Verified runtime: "
    f"torch={actual['torch']} torchvision={actual['torchvision']} "
    f"CUDA={actual['cuda_runtime']}"
)
PY
fi

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "sim:libero:libero" "${CAP_HARNESS_PROFILE:-rtx5090}"
cat <<EOF
Bootstrap complete.
  repository:    $REPO_ROOT
  environment:   $VENV_ROOT
  dependencies:  $REPO_ROOT/third_party
  LIBERO config: $LIBERO_CONFIG_DIR/config.yaml

Activate:
  source "$VENV_ROOT/bin/activate"
  export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

Start providers:
  scripts/supervise_services.sh

Diagnose:
  cap-harness doctor --reset --output validation-artifacts/doctor.json
EOF
