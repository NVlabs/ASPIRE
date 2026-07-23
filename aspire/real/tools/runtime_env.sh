#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


if [[ "${_RUNTIME_ENV_SH_SOURCED:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi
export _RUNTIME_ENV_SH_SOURCED=1

TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd -- "$TOOLS_DIR/.." && pwd)"
USER_RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-$HOME/.config/aspire/runtime_env.sh}"

sanitize_python_launcher_env() {
  unset VIRTUAL_ENV
  unset CONDA_PREFIX
  unset CONDA_DEFAULT_ENV
  unset PYTHONHOME
  unset __PYVENV_LAUNCHER__
}

warn_yellow() {
  printf '\033[33m[warn]\033[0m %s\n' "$*" >&2
}

path_prepend_once() {
  local dir="$1"
  local current="$2"
  if [[ -z "$dir" || ! -d "$dir" ]]; then
    printf '%s\n' "$current"
    return 0
  fi
  case ":$current:" in
    *":$dir:"*) printf '%s\n' "$current" ;;
    *) printf '%s\n' "${dir}${current:+:$current}" ;;
  esac
}

detect_tmp_root() {
  local candidate
  local candidates=(
    "${RUNTIME_TMP_ROOT:-}"
    "${TMPDIR:-}"
    "${XDG_CACHE_HOME:-}"
    "/var/tmp"
    "/tmp"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" ]]; then
      mkdir -p "$candidate" >/dev/null 2>&1 || true
      if [[ -d "$candidate" && -w "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  return 1
}

detect_hf_home() {
  local tmp_root="${1:-}"
  local candidate
  local candidates=(
    "${HF_HOME:-}"
    "${XDG_CACHE_HOME:-}/huggingface"
    "$HOME/.cache/huggingface"
    "${tmp_root:+$tmp_root/huggingface}"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate/hub" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

detect_cuda_home() {
  local candidate
  local candidates=(
    "${CUDA_HOME:-}"
    "/usr/local/cuda"
    "/usr/local/cuda-12.8"
    "/usr/local/cuda-12"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v nvcc >/dev/null 2>&1; then
    candidate="$(cd -- "$(dirname -- "$(dirname -- "$(command -v nvcc)")")" && pwd)"
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

source_user_bashrc() {
  if [[ -f "$HOME/.bashrc" ]]; then
    # shellcheck disable=SC1090
    if ! source "$HOME/.bashrc" >/dev/null 2>&1; then
      warn_yellow "failed to source ~/.bashrc; continuing with repo-relative defaults from $PROJECT_ROOT"
    fi
  fi
}

source_user_runtime_env_file() {
  if [[ -f "$USER_RUNTIME_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    if ! source "$USER_RUNTIME_ENV_FILE" >/dev/null 2>&1; then
      warn_yellow "failed to source $USER_RUNTIME_ENV_FILE"
    fi
  fi
}

export_repo_runtime_defaults() {
  export BUNDLESDF_REPO_LIB_DIR="$PROJECT_ROOT/third_party/bundlesdf/libs"
  export ANYGRASP_SERVICE_URL="${ANYGRASP_SERVICE_URL:-http://127.0.0.1:8122}"
  export CAP_CUROBO_REMOTE_PORT="${CAP_CUROBO_REMOTE_PORT:-8611}"
  export CAP_CUROBO_SSH_TARGET="${CAP_CUROBO_SSH_TARGET:-}"
  local preferred_hf_home=""

  if cuda_home="$(detect_cuda_home)"; then
    export CUDA_HOME="$cuda_home"
    export PATH="$(path_prepend_once "$CUDA_HOME/bin" "${PATH:-}")"
    export LD_LIBRARY_PATH="$(path_prepend_once "$CUDA_HOME/lib64" "${LD_LIBRARY_PATH:-}")"
  fi

  if tmp_root="$(detect_tmp_root)"; then
    export RUNTIME_TMP_ROOT="$tmp_root"
    export TMPDIR="${TMPDIR:-$RUNTIME_TMP_ROOT}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUNTIME_TMP_ROOT/.cache}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_TMP_ROOT/uv-cache}"
    preferred_hf_home="$(detect_hf_home "$RUNTIME_TMP_ROOT")"
    export HF_HOME="${HF_HOME:-$preferred_hf_home}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" >/dev/null 2>&1 || true
  else
    warn_yellow "could not determine a writable tmp/cache root; leaving TMPDIR/XDG_CACHE_HOME/UV_CACHE_DIR unchanged"
  fi
}

source_user_runtime_env_file
source_user_bashrc
sanitize_python_launcher_env
export_repo_runtime_defaults
