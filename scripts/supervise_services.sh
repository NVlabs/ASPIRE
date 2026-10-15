#!/usr/bin/env bash
# Start and supervise one shared, long-lived provider stack for a hardware
# profile. Providers run from their own per-provider virtual environments
# (.venv-<name>) and are reached over HTTP by the LIBERO and Robosuite
# simulator processes. Placement (GPU), ports, and bounded request
# concurrency come from the declarative topology + profile so there is a single
# source of truth. On one GPU this yields one provider instance with serialized
# request handling instead of a model loaded per simulator process.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Provider services locate their vendored checkpoints/assets through
# CAP_HARNESS_VENDOR_ROOT (e.g. Contact-GraspNet's checkpoints). Export it so
# each provider process resolves third_party/ correctly regardless of cwd.
export CAP_HARNESS_VENDOR_ROOT="${CAP_HARNESS_VENDOR_ROOT:-$REPO_ROOT/third_party}"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"
PROFILE="${CAP_HARNESS_PROFILE:-rtx5090}"
LOG_DIR="${CAP_HARNESS_SERVICE_LOG_DIR:-$REPO_ROOT/validation-artifacts/service-logs}"
READY_TIMEOUT_S="${CAP_HARNESS_SERVICE_READY_TIMEOUT_S:-300}"
SERVICE_HOST="${CAP_HARNESS_SERVICE_HOST:-127.0.0.1}"
PROVIDERS_CSV="${CAP_HARNESS_PROVIDERS:-sam3,pyroki,curobo}"

usage() {
  cat <<'EOF'
Usage: scripts/supervise_services.sh [--profile NAME] [--providers a,b,c]
                                     [--log-dir PATH] [--ready-timeout SEC]

Start the shared provider stack for a profile from configs/environments.json.
Each provider runs from its own <repo>/.venv-<name>. GPU placement, ports, and
bounded request concurrency are taken from the profile overlay.

Options:
  --profile NAME     Hardware profile (default: rtx5090).
  --providers CSV    Providers to start (default: sam3,pyroki,curobo).
  --log-dir PATH     Service log directory.
  --ready-timeout S  Maximum readiness wait (default: 300).

Deprecated (accepted with a warning during migration):
  --with-curobo      cuRobo is a normal provider; add it via --providers.
  --python PATH      Ignored; providers use their own per-provider venvs.
EOF
}

DEPRECATED_ADD=()
while (($#)); do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --providers) PROVIDERS_CSV="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --ready-timeout) READY_TIMEOUT_S="$2"; shift 2 ;;
    --with-curobo) printf 'warning: --with-curobo is deprecated; add curobo via --providers\n' >&2; DEPRECATED_ADD+=(curobo); shift ;;
    --python) printf 'warning: --python is deprecated and ignored; providers use per-provider venvs\n' >&2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
command -v curl >/dev/null 2>&1 || { printf 'error: curl is required\n' >&2; exit 1; }
[[ "$READY_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || { printf 'error: --ready-timeout must be a positive integer\n' >&2; exit 2; }
for extra in "${DEPRECATED_ADD[@]:-}"; do
  [[ -n "$extra" ]] && PROVIDERS_CSV="$PROVIDERS_CSV,$extra"
done

mkdir -p "$LOG_DIR"

# Resolve every requested provider through the single-source topology + profile.
# Emits: name|venv_basename|module|port|gpu_index|concurrency  (gpu_index blank => CPU)
readarray -t PLAN < <(
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" - "$REPO_ROOT" "$PROFILE" "$PROVIDERS_CSV" <<'PY'
import sys
from pathlib import Path
from cap_harness import environments

repo, profile_name, csv = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
topo = environments.load_topology(repo)
profile = environments.load_profile(profile_name, repo)
known = set(topo["providers"])
requested = [p.strip() for p in csv.split(",") if p.strip()]
unknown = [p for p in requested if p not in known]
if unknown:
    raise SystemExit(f"unknown provider(s): {', '.join(unknown)}; known: {sorted(known)}")
seen = set()
for name in requested:
    if name in seen:
        continue
    seen.add(name)
    spec = topo["providers"][name]
    r = environments.resolve_provider(name, profile, repo, topo)
    gpu = "" if r.gpu_index is None else str(r.gpu_index)
    print(f"{name}|{spec['venv']}|{spec.get('module','')}|{r.port}|{gpu}|{r.concurrency}")
PY
)
[[ "${#PLAN[@]}" -gt 0 ]] || { printf 'error: no providers resolved\n' >&2; exit 1; }

declare -a PIDS=() NAMES=() PORTS=()

shutdown() {
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap shutdown INT TERM EXIT

for row in "${PLAN[@]}"; do
  IFS='|' read -r name venv module port gpu conc <<< "$row"
  venv_python="$REPO_ROOT/$venv/bin/python"
  [[ -x "$venv_python" ]] || {
    printf 'error: provider %s venv is missing: %s (bootstrap it first)\n' "$name" "$venv_python" >&2
    exit 1
  }
  log_path="$LOG_DIR/$name.log"
  printf 'starting %s (venv=%s gpu=%s concurrency=%s port=%s log=%s)\n' \
    "$name" "$venv" "${gpu:-cpu}" "$conc" "$port" "$log_path"
  # --port matters: the port is resolved from the profile and used to
  # health-check below, but was never passed to the service, which fell back to
  # its own hardcoded default. Any profile setting "ports" would have polled one
  # port while the service listened on another. Invisible today only because
  # every shipped profile leaves "ports" empty.
  CUDA_VISIBLE_DEVICES="$gpu" CAP_HARNESS_SERVICE_CONCURRENCY="$conc" \
    "$venv_python" -m "$module" --port "$port" >"$log_path" 2>&1 &
  PIDS+=("$!"); NAMES+=("$name"); PORTS+=("$port")
done

wait_for_endpoint() {
  local name="$1" port="$2" pid="$3"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  while ((SECONDS < deadline)); do
    kill -0 "$pid" 2>/dev/null || { printf 'error: %s exited before ready; see %s/%s.log\n' "$name" "$LOG_DIR" "$name" >&2; return 1; }
    if curl --fail --silent --show-error --max-time 2 "http://$SERVICE_HOST:$port/openapi.json" >/dev/null; then
      printf 'ready: %s on %s:%s\n' "$name" "$SERVICE_HOST" "$port"; return 0
    fi
    sleep 2
  done
  printf 'error: %s did not become ready within %ss\n' "$name" "$READY_TIMEOUT_S" >&2; return 1
}

for i in "${!PIDS[@]}"; do wait_for_endpoint "${NAMES[$i]}" "${PORTS[$i]}" "${PIDS[$i]}"; done

printf 'all provider services ready (profile %s); supervising in foreground\n' "$PROFILE"
while true; do
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      status=0; wait "${PIDS[$i]}" || status=$?
      printf 'error: %s exited unexpectedly with status %s\n' "${NAMES[$i]}" "${status:-0}" >&2
      exit 1
    fi
  done
  sleep 2
done
