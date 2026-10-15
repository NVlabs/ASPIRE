# Shared helper: per-venv bootstrap fingerprints for no-op re-bootstraps.
#
# A fingerprint hashes the inputs that actually determine a venv's contents:
# the resolved dependency set (uv.lock + pyproject.toml), the build architecture
# (profile -> gpu_runtime.arches), the environment marker (name + extra), and the
# pinned commits of any submodules the environment builds from. It is written to
# <venv>/.cap-fingerprint.json after a successful bootstrap. If the fingerprint
# is unchanged the bootstrap is a genuine no-op instead of a full rebuild.

# compute_venv_fingerprint <repo> <profile> <marker> [submodule ...] -> prints hash
compute_venv_fingerprint() {
  local repo="$1" profile="$2" marker="$3"; shift 3
  local arch
  arch="$(PYTHONPATH="$repo/src" python3 - "$repo" "$profile" <<'PY' 2>/dev/null || echo noarch
import sys
from pathlib import Path
from cap_harness import environments
repo, name = Path(sys.argv[1]), sys.argv[2]
p = environments.load_profile(name, repo)
print(p["arch"] + ":" + environments.cuda_arch_list(p["arch"], repo))
PY
)"
  {
    sha256sum "$repo/uv.lock" 2>/dev/null | awk '{print $1}'
    sha256sum "$repo/pyproject.toml" 2>/dev/null | awk '{print $1}'
    sha256sum "$repo/configs/dependency-lock.json" 2>/dev/null | awk '{print $1}'
    printf 'marker=%s\narch=%s\n' "$marker" "$arch"
    local sub
    for sub in "$@"; do
      printf 'sub=%s@%s\n' "$sub" "$(git -C "$repo" ls-files --stage -- "$sub" 2>/dev/null | awk '{print $2}')"
    done
  } | sha256sum | awk '{print $1}'
}

# venv_fingerprint_current <venv_path> <fingerprint> -> 0 if venv exists and matches
venv_fingerprint_current() {
  local venv="$1" fp="$2" f="$1/.cap-fingerprint.json"
  [[ -x "$venv/bin/python" && -f "$f" ]] || return 1
  local cur
  cur="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("fingerprint",""))' "$f" 2>/dev/null || echo)"
  [[ -n "$cur" && "$cur" == "$fp" ]]
}

# write_venv_fingerprint <venv_path> <fingerprint> <marker> <profile>
write_venv_fingerprint() {
  python3 - "$1/.cap-fingerprint.json" "$2" "$3" "$4" <<'PY'
import json, sys
path, fp, marker, profile = sys.argv[1:5]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"schema": "cap-harness/fingerprint/1", "fingerprint": fp, "marker": marker, "profile": profile},
        handle, indent=2, sort_keys=True,
    )
    handle.write("\n")
PY
}
