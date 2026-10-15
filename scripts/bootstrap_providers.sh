#!/usr/bin/env bash
# Selective, orchestration-only aggregator: bootstrap a chosen subset of the
# provider stack for a hardware profile. This script installs nothing itself; it
# dispatches to the per-provider bootstraps in order. There is deliberately no
# bootstrap_sim.sh mega-installer -- simulators are bootstrapped independently.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-python3}"
PROFILE="${CAP_HARNESS_PROFILE:-rtx5090}"
PROVIDERS_CSV=""

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_providers.sh --providers a,b,c [--profile NAME]

Orchestration-only aggregator. Dispatches to the per-provider bootstrap for
each named provider. Provider names are keys under "providers" in
configs/environments.json (sam3, pyroki, contact_graspnet, curobo).

Examples:
  scripts/bootstrap_providers.sh --providers sam3,pyroki,curobo --profile rtx5090
  scripts/bootstrap_providers.sh --providers sam3,pyroki,curobo --profile l40
EOF
}

while (($#)); do
  case "$1" in
    --providers) PROVIDERS_CSV="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$PROVIDERS_CSV" ]] || { printf 'error: --providers is required\n' >&2; usage >&2; exit 2; }

# Validate the profile and every provider name against the topology up front so a
# typo fails before any environment is built.
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" - "$REPO_ROOT" "$PROFILE" "$PROVIDERS_CSV" <<'PY'
import sys
from pathlib import Path
from cap_harness import environments

repo, profile_name, csv = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
environments.load_profile(profile_name, repo)
known = set(environments.provider_names(repo))
requested = [p.strip() for p in csv.split(",") if p.strip()]
unknown = [p for p in requested if p not in known]
if unknown:
    raise SystemExit(f"unknown provider(s): {', '.join(unknown)}; known: {sorted(known)}")
if not requested:
    raise SystemExit("no providers requested")
PY

IFS=',' read -r -a PROVIDERS <<< "$PROVIDERS_CSV"
printf 'Bootstrapping providers [%s] for profile %s\n' "$PROVIDERS_CSV" "$PROFILE"
for provider in "${PROVIDERS[@]}"; do
  provider="${provider// /}"
  [[ -n "$provider" ]] || continue
  printf '\n=== provider: %s ===\n' "$provider"
  CAP_HARNESS_PROFILE="$PROFILE" "$REPO_ROOT/scripts/bootstrap_provider.sh" "$provider" --profile "$PROFILE"
done
printf '\nAll requested providers bootstrapped for profile %s.\n' "$PROFILE"
