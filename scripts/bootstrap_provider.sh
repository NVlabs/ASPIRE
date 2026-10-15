#!/usr/bin/env bash
# Topology-driven bootstrap for a single pip-based provider (sam3, pyroki,
# contact_graspnet). Reads configs/environments.json for the venv basename,
# extra, service module, and submodule so there is one source of truth. Source-
# built providers (curobo) have their own specialized scripts and are
# delegated to here rather than reimplemented.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"
PROFILE="${CAP_HARNESS_PROFILE:-rtx5090}"
PROVIDER=""

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_provider.sh <provider> [--profile NAME]

Bootstrap one provider's virtual environment from the declarative topology.
<provider> is a key under "providers" in configs/environments.json
(e.g. sam3, pyroki, contact_graspnet). Source-built providers (curobo) are
delegated to their specialized scripts.
EOF
}

while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *) if [[ -n "$PROVIDER" ]]; then printf 'error: one provider at a time\n' >&2; exit 2; fi; PROVIDER="$1"; shift ;;
  esac
done
[[ -n "$PROVIDER" ]] || { usage >&2; exit 2; }
command -v uv >/dev/null || { printf 'error: uv is required\n' >&2; exit 1; }

# Resolve the provider spec + profile from the single-source topology.
readarray -t SPEC < <(
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" - "$REPO_ROOT" "$PROVIDER" "$PROFILE" <<'PY'
import sys
from pathlib import Path
from cap_harness import environments

repo, name, profile_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
topo = environments.load_topology(repo)
if name not in topo["providers"]:
    raise SystemExit(f"unknown provider {name!r}; known: {sorted(topo['providers'])}")
spec = topo["providers"][name]
profile = environments.load_profile(profile_name, repo)
resolved = environments.resolve_provider(name, profile, repo, topo)
print(spec["venv"])
print(spec.get("extra", ""))
print(spec.get("module", ""))
print(spec.get("submodule", ""))
print(spec.get("import_check", ""))
print("1" if spec.get("source_build") else "0")
print(resolved.cuda_arch_list or "")
PY
)
VENV_BASENAME="${SPEC[0]}"
EXTRA="${SPEC[1]}"
MODULE="${SPEC[2]}"
SUBMODULE="${SPEC[3]}"
IMPORT_CHECK="${SPEC[4]}"
SOURCE_BUILD="${SPEC[5]}"
CUDA_ARCH_LIST="${SPEC[6]}"

if [[ "$SOURCE_BUILD" == "1" ]]; then
  printf 'provider %s is source-built; delegating to scripts/bootstrap_%s.sh\n' "$PROVIDER" "$PROVIDER"
  export CAP_HARNESS_PROFILE="$PROFILE"
  exec "$REPO_ROOT/scripts/bootstrap_${PROVIDER}.sh" "$@"
fi

# Canonical repo-local venv path; CAP_HARNESS_VENV_ROOT backing is applied by
# resolve_venv_dir (shared helper) so shared node-local storage stays behind
# a stable <repo>/.venv-* symlink.
VENV_ROOT="$(resolve_canonical_venv "$REPO_ROOT" "$VENV_BASENAME")"
UV_CACHE_ROOT="${CAP_HARNESS_PROVIDER_CACHE:-$REPO_ROOT/.uv-cache}"

# No-op re-bootstrap: if the resolved deps, arch, and submodule pin are unchanged
# from the recorded fingerprint, skip the rebuild entirely.
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "$PROFILE" "provider:$PROVIDER:$EXTRA" ${SUBMODULE:+"$SUBMODULE"})"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf 'provider %s is up to date (fingerprint %s); skipping rebuild\n' "$PROVIDER" "${FINGERPRINT:0:12}"
  exit 0
fi

if [[ -n "$SUBMODULE" ]]; then
  git -C "$REPO_ROOT" submodule update --init "$SUBMODULE"
  expected="$(git -C "$REPO_ROOT" ls-files --stage -- "$SUBMODULE" | awk '{print $2}')"
  actual="$(git -C "$REPO_ROOT/$SUBMODULE" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || {
    printf 'error: %s checkout mismatch: %s != %s\n' "$SUBMODULE" "$actual" "$expected" >&2
    exit 1
  }
fi

mkdir -p "$UV_CACHE_ROOT"
# Preserve an existing environment (idempotent bootstrap); Stage 8 adds a
# fingerprint short-circuit so an unchanged bootstrap is a genuine no-op.
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
VIRTUAL_ENV="$VENV_ROOT" UV_PROJECT_ENVIRONMENT="$VENV_ROOT" UV_CACHE_DIR="$UV_CACHE_ROOT" \
  uv sync --frozen --active --project "$REPO_ROOT" --extra "$EXTRA"

"$VENV_ROOT/bin/python" - "$VENV_BASENAME" "$IMPORT_CHECK" "$MODULE" <<'PY'
import importlib
import sys
from pathlib import Path

basename, import_check, module = sys.argv[1:4]
root = Path(sys.prefix).resolve()
assert root.name == basename, f"{root} != {basename}"
import cap_harness  # noqa: F401
if import_check:
    importlib.import_module(import_check)
print(f"Verified provider env: {root} ({import_check or module})")
PY

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "provider:$PROVIDER:$EXTRA" "$PROFILE"
printf 'Provider bootstrap complete: %s -> %s (profile %s)\n' "$PROVIDER" "$VENV_ROOT" "$PROFILE"
