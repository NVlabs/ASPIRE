# Shared helper: canonical <repo>/.venv-* paths with optional node-local backing.
#
# Canonical, user-facing venv paths are always <repo>/.venv-<name>. On local
# storage these are real directories. When CAP_HARNESS_VENV_ROOT is set (e.g.
# shared node-local storage), the real environment lives at
# $CAP_HARNESS_VENV_ROOT/.venv-<name> and <repo>/.venv-<name> is a symlink to it,
# so the canonical path is stable regardless of where the bytes live.

# resolve_canonical_venv <repo_root> <basename>  -> prints the canonical path
resolve_canonical_venv() {
  local repo_root="$1" basename="$2"
  local canonical="$repo_root/$basename"
  local root="${CAP_HARNESS_VENV_ROOT:-}"
  if [[ -n "$root" ]]; then
    local backing="$root/$basename"
    mkdir -p "$backing"
    if [[ -L "$canonical" ]]; then
      local current; current="$(readlink -- "$canonical")"
      if [[ "$current" != "$backing" ]]; then
        rm -f -- "$canonical"; ln -s -- "$backing" "$canonical"
      fi
    elif [[ -e "$canonical" ]]; then
      printf 'error: %s exists and is not a symlink; move it aside to use CAP_HARNESS_VENV_ROOT backing\n' "$canonical" >&2
      return 1
    else
      ln -s -- "$backing" "$canonical"
    fi
  fi
  printf '%s\n' "$canonical"
}
