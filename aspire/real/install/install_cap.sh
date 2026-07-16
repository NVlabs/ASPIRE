#!/usr/bin/env bash
# install_cap.sh — One-shot installer for the CAP family of services.
#
# Run from the repo root:  bash install/install_cap.sh
#
# Phases run in strict dependency order. Re-running is idempotent.
#
#   1. Recovered dependency source preflight
#        - creates metadata-only stubs for optional RoboSuite/RoboCasa extras
#        - requires reviewed i2rt, PyRoki, BundleSDF, and cuRobo sources
#        - never pulls private Git LFS or vendor authorization payloads
#
#   2. Miniforge + aspire-anygrasp-libs conda env [moved here so PyAudio's
#      C build in step 3 can find portaudio.h, libportaudio.so]
#        - installs miniforge to ~/miniforge3 if missing
#        - creates env: openssl=1.1, openblas, portaudio
#
#   3. Python venv (uv sync) — single uv sync covering cap, cap_tools, stt
#        - All deps come from pyproject.toml [project.optional-dependencies]:
#          nvidia-curobo (editable, no-build-isolation), graspnetAPI, ninja
#          live under cap_tools; transforms3d>=0.4 enforced via
#          [tool.uv] override-dependencies (defeats graspnetAPI's ==0.3.1 pin).
#        - PyAudio's source build picks up portaudio from the conda env via
#          CPATH / LIBRARY_PATH set just for this step.
#        - nvidia-curobo builds inline (CUDA_HOME + setuptools-scm pretend).
#
#   4. (removed — handled in phase 3 via lockfile)
#
#   5. Hugging Face models (gated) — facebook/sam3, facebook/sam2.1-hiera-large
#
#   6. cuRobo verify (already installed in phase 3 via lockfile)
#
#   7. BundleSDF — delegates to install/compile_bundlesdf.sh
#        - that script bootstraps a separate bundlesdf-build conda env,
#          builds OpenCV-CUDA + BundleTrack, and writes runtime closure to
#          third_party/bundlesdf_5090/ (kept SEPARATE from the LFS-tracked
#          third_party/bundlesdf/libs/, which is the 4070/4090 variant)
#
#   7.5 ZED SDK (auto, if USB device present) — delegates to install/build_zed_sdk.sh
#        - matches the native SDK major.minor to the pyzed pin in pyproject.toml
#        - re-installs the pyzed wheel against the new SDK with --no-deps
#          (won't disturb numpy/cython/anything else)
#        - skipped if lsusb shows no Stereolabs device; force with INSTALL_ZED_SDK=1
#
#   8. Extract runtime libs into third_party/anygrasp_libs/
#        - libssl.so.1.1, libcrypto.so.1.1, libopenblas.so.0, libportaudio.so
#          plus their transitive .so closure
#        - all from aspire-anygrasp-libs (created in phase 2)
#
#   9. Node.js prebuilt → third_party/nodejs/
#
#  10. Identity + API key secrets (zenity popup or terminal prompt)
#        - CAP_AGENT_NAME       — name shown in cap_ui
#        - NVIDIA_API_KEY       — default VLM backend (vision queries)
#        - GEMINI_API_KEY       — optional, Gemini backend (cap/agent/reflection.py)
#        - ANTHROPIC_API_KEY    — optional, Claude backend
#        - OPENAI_API_KEY       — optional, GPT backend
#        - ELEVENLABS_API_KEY   — optional, voice output via ElevenLabs
#        - ELEVENLABS_VOICE_ID  — required iff ELEVENLABS_API_KEY is set
#        All persisted to <repo>/.forge_env.secrets (gitignored, chmod 600,
#        preserved across reinstalls). The launcher sources .forge_env which
#        in turn sources .forge_env.secrets, so every pane / nohup'd service
#        process (bridge, voice, cap_agent, cap_server) inherits these.
#
#  11. .forge_env.runtime — generated runtime paths sourced by the operator's
#      .forge_env; sets ANYGRASP_OPENSSL11_DIR, OPENBLAS_HOME,
#      BUNDLESDF_RUNTIME_LIB_DIR, LD_LIBRARY_PATH, and PATH.
#
# Does NOT touch:
#   - any path under /usr or /lib
#   - /usr/local/cuda* (read-only)
#   - kernel modules, NVIDIA driver, nvidia-smi
#
# After install, conda envs at ~/miniforge3/envs/{bundlesdf-build,aspire-anygrasp-libs}/
# are optional cache; runtime depends only on .venv/ and third_party/.
#
# See install/README.md for the supported workflow and remaining blockers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
AG_ENV="$CONDA_BASE/envs/aspire-anygrasp-libs"
ANYGRASP_LIBS_DIR="$REPO_ROOT/third_party/anygrasp_libs"
SECRETS_FILE="$REPO_ROOT/.forge_env.secrets"
ENV_SNIPPET="$REPO_ROOT/.forge_env.runtime"
BUNDLESDF_LOCK_DIR="$SCRIPT_DIR/locks/bundlesdf"
BUNDLESDF_LOCK_FILE="$BUNDLESDF_LOCK_DIR/build-lock.env"
[[ -f "$BUNDLESDF_LOCK_FILE" ]] || { echo "[install] missing $BUNDLESDF_LOCK_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$BUNDLESDF_LOCK_FILE"

for command_name in wget sha256sum uv; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "[install] ERROR: required command is unavailable: $command_name" >&2
        exit 1
    }
done

download_verified() {
    local url="$1" expected="$2" output="$3"
    wget -q --show-progress -O "$output" "$url"
    printf '%s  %s\n' "$expected" "$output" | sha256sum -c -
}

# Source any previously-saved secrets (HF_TOKEN, GEMINI_API_KEY, etc.) so
# downstream phases (5 = HF model downloads, etc.) can use them without
# re-prompting. The phase 10 prompt loop re-saves anything still missing.
# shellcheck disable=SC1090
[[ -f "$SECRETS_FILE" ]] && source "$SECRETS_FILE"

# ─── Phase 1: recovered dependency sources ─────────────────────────────────
echo "[install] phase 1: local dependency sources"
# uv.lock references third_party/robosuite as an editable source. uv only
# reads its pyproject.toml's name+version to verify the lock — it doesn't
# actually install robosuite unless `--extra robocasa` is requested. To
# avoid a slow ~500 MB git clone of robosuite's asset tree on first install,
# we drop in a 4-line STUB pyproject.toml. If you later need the real source
# (e.g. you're using --extra robocasa or developing against it), run:
#   rm -rf third_party/robosuite && git submodule update --init third_party/robosuite
if [[ ! -f "$REPO_ROOT/third_party/robosuite/pyproject.toml" ]]; then
    echo "[install]   stubbing third_party/robosuite/pyproject.toml (skip ~500 MB clone)"
    mkdir -p "$REPO_ROOT/third_party/robosuite"
    cat > "$REPO_ROOT/third_party/robosuite/pyproject.toml" <<'PROJ'
# Stub created by install/install_cap.sh to satisfy uv's editable-path metadata
# check without cloning the full robosuite asset tree. Replace with a real
# checkout from the public upstream before installing the robocasa extra or
# actually using robosuite at runtime; see patches/dependencies/README.md.
[project]
name = "robosuite"
version = "1.5.2"
requires-python = ">=3.10"
PROJ
fi
echo "[install]   robosuite stub: OK"

if [[ ! -f "$REPO_ROOT/third_party/robocasa/pyproject.toml" && ! -f "$REPO_ROOT/third_party/robocasa/setup.py" ]]; then
    echo "[install]   stubbing optional third_party/robocasa metadata"
    mkdir -p "$REPO_ROOT/third_party/robocasa"
    cat > "$REPO_ROOT/third_party/robocasa/pyproject.toml" <<'PROJ'
[project]
name = "robocasa"
version = "1.0.0"
requires-python = ">=3.10"
PROJ
fi

for d in third_party/i2rt third_party/pyroki third_party/bundlesdf third_party/curobo; do
    if [ ! -f "$d/pyproject.toml" ] && [ ! -f "$d/setup.py" ]; then
        echo "[install] ERROR: $d is missing." >&2
        echo "[install] Recover reviewed sources and patches using patches/dependencies/README.md." >&2
        exit 2
    fi
done
if [[ ! -d third_party/anygrasp_sdk ]]; then
    echo "[install] WARNING: third_party/anygrasp_sdk is absent; AnyGrasp remains unavailable." >&2
fi

# ─── Phase 2: miniforge + aspire-anygrasp-libs (openssl 1.1, openblas, portaudio) ───
echo "[install] phase 2: miniforge + aspire-anygrasp-libs conda env"
if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
    echo "[install]   installing miniforge to $CONDA_BASE..."
    INSTALLER="$(mktemp --suffix=.sh)"
    download_verified "$MINIFORGE_URL" "$MINIFORGE_SHA256" "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_BASE"
    rm -f "$INSTALLER"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -d "$AG_ENV" ]]; then
    echo "[install]   creating aspire-anygrasp-libs (openssl=1.1 + openblas + portaudio)..."
    conda create -y -n aspire-anygrasp-libs -c conda-forge \
        'openssl=1.1' openblas portaudio
elif [[ ! -e "$AG_ENV/lib/libssl.so.1.1" || ! -e "$AG_ENV/lib/libportaudio.so" ]]; then
    # Pass all three specs together so conda doesn't drift openssl up to 3.x
    # when adding portaudio (or vice versa).
    echo "[install]   reconciling aspire-anygrasp-libs (openssl 1.1 + openblas + portaudio)..."
    conda install -y -n aspire-anygrasp-libs -c conda-forge \
        'openssl=1.1' openblas portaudio
else
    echo "[install]   aspire-anygrasp-libs already has openssl=1.1 + portaudio; reusing"
fi
[[ -e "$AG_ENV/include/portaudio.h" ]] || { echo "[install] ERROR: portaudio.h missing in $AG_ENV/include" >&2; exit 1; }
[[ -e "$AG_ENV/lib/libportaudio.so" ]] || { echo "[install] ERROR: libportaudio.so missing in $AG_ENV/lib"   >&2; exit 1; }
[[ -e "$AG_ENV/lib/libssl.so.1.1"   ]] || { echo "[install] ERROR: libssl.so.1.1 missing in $AG_ENV/lib"     >&2; exit 1; }

# ─── Phase 3: Python venv (uv sync with all extras) ────────────────────────
# Single uv sync handles everything declared in pyproject.toml:
#   • cap_tools brings in nvidia-curobo (editable path; no-build-isolation),
#     graspnetAPI, ninja — see [tool.uv.sources] and [tool.uv].
#   • [tool.uv] override-dependencies relaxes graspnetAPI's transforms3d pin.
# Env vars below:
#   • CPATH / LIBRARY_PATH — PyAudio (--extra stt) needs portaudio.h to build.
#   • CUDA_HOME / SETUPTOOLS_SCM_… — nvidia-curobo build prereqs (host-CUDA).
echo "[install] phase 3: uv sync (cap, cap_tools, stt — incl. cuRobo + graspnetAPI)"
# nvidia-curobo uses no-build-isolation, so its editable build runs `import
# torch` against the venv — but on a fresh venv torch isn't installed yet,
# so the metadata-generation step fails. Two-pass sync: first pass skips
# curobo (installing torch via base deps), second pass builds curobo with
# torch present. Versions stay locked to uv.lock — no manual pin needed.
CPATH="$AG_ENV/include${CPATH:+:$CPATH}" \
LIBRARY_PATH="$AG_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
    uv sync --extra cap --extra cap_tools --extra stt --no-install-package nvidia-curobo
CPATH="$AG_ENV/include${CPATH:+:$CPATH}" \
LIBRARY_PATH="$AG_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
    uv sync --extra cap --extra cap_tools --extra stt

# ─── Phase 5: Hugging Face models ──────────────────────────────────────────
echo "[install] phase 5: Hugging Face auth + model downloads"
if [[ "${SKIP_SAM_MODELS:-0}" == "1" || "${SKIP_HF_MODELS:-0}" == "1" ]]; then
    echo "[install]   SKIP_SAM_MODELS/SKIP_HF_MODELS set — skipping facebook/sam3 and facebook/sam2.1-hiera-large downloads"
else
# Resolution order for the HF token:
#   1. HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var (set by sourcing .forge_env.secrets
#      at the top of this script, or by the user externally)
#   2. cached login at ~/.cache/huggingface/token (`hf auth whoami` finds it)
#   3. interactive prompt (zenity popup if DISPLAY available, else terminal `read -s`)
#      — saved to .forge_env.secrets for future installs.
if [[ -z "${HF_TOKEN:-}${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    HF_USER="$(uv run hf auth whoami 2>/dev/null | awk '/^user:|^[Uu]ser/ {print $NF; exit}')"
    if [[ -n "$HF_USER" && "$HF_USER" != "Not" ]]; then
        echo "[install]   HF cached login: $HF_USER (using ~/.cache/huggingface/token)"
    else
        echo "[install]   no HF token in env or cache — prompting..."
        if [[ -n "${DISPLAY:-}" ]] && command -v zenity >/dev/null 2>&1; then
            HF_TOKEN_INPUT=$(zenity --entry --hide-text \
                --title="CAP Install — HF_TOKEN" \
                --text="HuggingFace token (https://huggingface.co/settings/tokens; READ scope is fine):" 2>/dev/null || true)
        else
            read -rs -p "  HF_TOKEN (paste from https://huggingface.co/settings/tokens; READ scope): " HF_TOKEN_INPUT; echo
        fi
        if [[ -n "$HF_TOKEN_INPUT" ]]; then
            touch "$SECRETS_FILE"; chmod 600 "$SECRETS_FILE"
            # Replace any existing HF_TOKEN line then append
            sed -i '/^export HF_TOKEN=/d' "$SECRETS_FILE"
            printf 'export HF_TOKEN=%q\n' "$HF_TOKEN_INPUT" >> "$SECRETS_FILE"
            export HF_TOKEN="$HF_TOKEN_INPUT"
            echo "[install]   saved HF_TOKEN to .forge_env.secrets and exported for this run"
        else
            echo "[install]   WARNING: no HF token provided; sam3 (gated) will fail to download" >&2
        fi
    fi
else
    echo "[install]   HF_TOKEN already set in env (from .forge_env.secrets)"
fi
: "${HF_DOWNLOAD_MAX_WORKERS:=16}"
: "${HF_XET_HIGH_PERFORMANCE:=1}"
: "${HF_XET_NUM_CONCURRENT_RANGE_GETS:=32}"
: "${HF_XET_MAX_CONCURRENT_DOWNLOADS:=16}"
export HF_XET_HIGH_PERFORMANCE HF_XET_NUM_CONCURRENT_RANGE_GETS HF_XET_MAX_CONCURRENT_DOWNLOADS
echo "[install]   HF download acceleration: max-workers=$HF_DOWNLOAD_MAX_WORKERS, HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE, HF_XET_NUM_CONCURRENT_RANGE_GETS=$HF_XET_NUM_CONCURRENT_RANGE_GETS"

echo "[install]   downloading facebook/sam3 (model.safetensors + Transformers metadata; skipping sam3.pt) ..."
if ! uv run hf download --max-workers "$HF_DOWNLOAD_MAX_WORKERS" facebook/sam3 \
    model.safetensors \
    config.json \
    processor_config.json \
    tokenizer_config.json \
    special_tokens_map.json \
    tokenizer.json \
    vocab.json \
    merges.txt; then
    echo "[install]   WARNING: facebook/sam3 download failed." >&2
    echo "[install]            Accept the license at https://huggingface.co/facebook/sam3 — token alone isn't enough." >&2
fi

echo "[install]   downloading facebook/sam2.1-hiera-large ..."
if ! uv run hf download --max-workers "$HF_DOWNLOAD_MAX_WORKERS" facebook/sam2.1-hiera-large; then
    echo "[install]   WARNING: facebook/sam2.1-hiera-large download failed." >&2
    echo "[install]            Accept the license at https://huggingface.co/facebook/sam2.1-hiera-large — token alone isn't enough." >&2
fi
fi

# ─── Phase 6: cuRobo verify (installed in phase 3 via lockfile) ────────────
echo "[install] phase 6: cuRobo verify"
if .venv/bin/python - <<'PY' >/dev/null 2>&1
import importlib.util, sys, torch, curobo  # noqa: F401
sys.exit(0 if (torch.cuda.is_available() and importlib.util.find_spec("curobo.geom") is not None) else 1)
PY
then
    echo "[install]   cuRobo OK ✓"
else
    echo "[install]   ERROR: cuRobo failed to import — see phase 3 uv sync output above." >&2
    exit 1
fi

# ─── Phase 7: BundleSDF (reuse a local build or compile from locked inputs) ───
echo "[install] phase 7: BundleSDF runtime libs"
BSDF_5090="$REPO_ROOT/third_party/bundlesdf_5090"
BSDF_MARKER="$BSDF_5090/libBundleTrack.so"
is_lfs_pointer() { [[ -f "$1" ]] && head -c 60 "$1" 2>/dev/null | grep -q '^version https://git-lfs'; }

# Step 1: smoke-test an existing local build to confirm it loads on this host.
# Existing builds may still fail on a host with a different Python, CUDA, or
# system ABI, so always test the import before reuse.
PREBUILT_OK=false
if [[ -f "$BSDF_MARKER" ]] && ! is_lfs_pointer "$BSDF_MARKER"; then
    if BUNDLESDF_RUNTIME_LIB_DIR="$BSDF_5090" LD_LIBRARY_PATH="$BSDF_5090:${LD_LIBRARY_PATH:-}" \
       .venv/bin/python -c "import bundlesdf" >/dev/null 2>&1; then
        PREBUILT_OK=true
        echo "[install]   prebuilt third_party/bundlesdf_5090/ loads cleanly — skipping rebuild ✓"
    else
        echo "[install]   prebuilt third_party/bundlesdf_5090/ exists but won't load on this host"
    fi
fi

# Step 2: if the local build isn't usable, build from locked source inputs.
if ! $PREBUILT_OK; then
    echo "[install]   compiling BundleSDF from source via $SCRIPT_DIR/compile_bundlesdf.sh"
    bash "$SCRIPT_DIR/compile_bundlesdf.sh"
fi

# ─── Phase 7.5: ZED SDK (only when a ZED is attached, or forced) ─────────
# pyzed wheel is already installed by phase 3 (pinned in pyproject.toml).
# What's missing for hosts with a ZED 2i is the native SDK at /usr/local/zed.
# Delegated to install/build_zed_sdk.sh which:
#   - matches the SDK major.minor to the pyzed pin in pyproject.toml
#   - runs the .run installer as the regular user (zenity-popup for sudo when DISPLAY is set)
#   - finishes with `uv pip install --no-deps <pyzed-url>` so nothing else moves
#
# Toggle:  INSTALL_ZED_SDK=1 forces install.
#          INSTALL_ZED_SDK=0 forces skip.
#          Unset (default): install only if `lsusb` sees a Stereolabs device.
echo "[install] phase 7.5: ZED SDK (gated by USB detection / INSTALL_ZED_SDK)"
ZED_DETECTED=0
command -v lsusb >/dev/null 2>&1 && lsusb 2>/dev/null | grep -qi stereolabs && ZED_DETECTED=1
WANT_ZED="${INSTALL_ZED_SDK:-$ZED_DETECTED}"
if [[ "$WANT_ZED" == "1" ]]; then
    echo "[install]   ZED present (or INSTALL_ZED_SDK=1) — running install/build_zed_sdk.sh"
    bash "$SCRIPT_DIR/build_zed_sdk.sh"
else
    echo "[install]   no ZED detected via lsusb and INSTALL_ZED_SDK unset — skipping SDK install"
    echo "[install]   (pyzed wheel is still installed in .venv via phase 3; SDK native libs missing)"
    echo "[install]   to install later:  bash install/build_zed_sdk.sh"
fi

# ─── Phase 8: Extract AnyGrasp + portaudio runtime libs ────────────────────
echo "[install] phase 8: extracting runtime libs into $ANYGRASP_LIBS_DIR"
mkdir -p "$ANYGRASP_LIBS_DIR"
for lib in libssl.so.1.1 libcrypto.so.1.1 libopenblas.so.0 libportaudio.so; do
    [[ -e "$AG_ENV/lib/$lib" ]] || { echo "[install]   ERROR: missing $AG_ENV/lib/$lib" >&2; exit 1; }
    cp -L --no-preserve=ownership "$AG_ENV/lib/$lib" "$ANYGRASP_LIBS_DIR/"
done
# PyAudio at runtime opens libportaudio.so.2; symlink to the unversioned conda binary.
ln -sf libportaudio.so "$ANYGRASP_LIBS_DIR/libportaudio.so.2"
# Transitive .so closure (e.g. libgfortran for libopenblas).
declare -A AG_SEEN
for f in "$ANYGRASP_LIBS_DIR"/*.so*; do AG_SEEN[$(basename "$f")]=1; done
ag_queue=("$ANYGRASP_LIBS_DIR"/*.so*)
while [[ ${#ag_queue[@]} -gt 0 ]]; do
    cur="${ag_queue[0]}"; ag_queue=("${ag_queue[@]:1}")
    while IFS= read -r line; do
        path=$(awk '{ for (i=1;i<=NF;i++) if ($i=="=>") { print $(i+1); exit } }' <<<"$line")
        [[ -n "$path" && "$path" != "not" && -f "$path" ]] || continue
        [[ "$path" == "$AG_ENV"/* ]] || continue
        bn=$(basename "$path")
        [[ -z "${AG_SEEN[$bn]:-}" ]] || continue
        AG_SEEN[$bn]=1
        cp -L --no-preserve=ownership "$path" "$ANYGRASP_LIBS_DIR/$bn"
        ag_queue+=("$ANYGRASP_LIBS_DIR/$bn")
    done < <(LD_LIBRARY_PATH="$ANYGRASP_LIBS_DIR" ldd "$cur" 2>/dev/null || true)
done
# Unversioned SONAME symlinks (libssl.so → libssl.so.1.1, etc.)
( cd "$ANYGRASP_LIBS_DIR" && for f in *.so.*; do
    base="${f%.so.*}.so"
    [[ ! -e "$base" ]] && ln -sf "$f" "$base"
done ) || true
echo "[install]   $ANYGRASP_LIBS_DIR has $(ls "$ANYGRASP_LIBS_DIR" | wc -l) entries"

# ─── Phase 9: Node.js prebuild ─────────────────────────────────────────────
NODE_VER="${ASPIRE_NODE_VER:-22.11.0}"
NODE_DIR="$REPO_ROOT/third_party/nodejs"
echo "[install] phase 9: nodejs $NODE_VER"
if [[ ! -x "$NODE_DIR/bin/node" ]]; then
    if [[ -z "${NODE_SHA256:-}" ]]; then
        echo "[install]   NODE_SHA256 is unset; skipping optional Node.js runtime." >&2
        echo "[install]   Set NODE_SHA256 to the official archive checksum to install it." >&2
    else
    NODE_TGZ="$(mktemp --suffix=.tar.xz)"
    NODE_URL="https://nodejs.org/dist/v${NODE_VER}/node-v${NODE_VER}-linux-x64.tar.xz"
    download_verified "$NODE_URL" "$NODE_SHA256" "$NODE_TGZ"
    mkdir -p "$NODE_DIR"
    tar -xJf "$NODE_TGZ" -C "$NODE_DIR" --strip-components=1
    rm -f "$NODE_TGZ"
    echo "[install]   nodejs ready ($("$NODE_DIR/bin/node" --version))"
    fi
else
    echo "[install]   already present ($("$NODE_DIR/bin/node" --version))"
fi

# ─── Phase 10: Voice / agent secrets (zenity popup, falls back to terminal) ───
echo "[install] phase 10: voice + agent identity secrets"
touch "$SECRETS_FILE"
chmod 600 "$SECRETS_FILE"

prompt_secret() {
    # $1 = env var name, $2 = label, $3 = "1" for hidden input (passwords)
    local var="$1" label="$2" hidden="${3:-0}"
    if grep -q "^export $var=" "$SECRETS_FILE" 2>/dev/null; then
        echo "[install]   $var already set in .forge_env.secrets — skipping prompt"
        return
    fi
    local value="${!var:-}"
    if [[ -z "$value" ]]; then
        if [[ -n "${DISPLAY:-}" ]] && command -v zenity >/dev/null 2>&1; then
            local zenflags=(--entry --title="CAP Install — $var" --text="$label")
            [[ "$hidden" == "1" ]] && zenflags+=(--hide-text)
            value=$(zenity "${zenflags[@]}" 2>/dev/null || true)
        elif [[ -t 0 ]]; then
            if [[ "$hidden" == "1" ]]; then
                read -rs -p "  $label: " value; echo
            else
                read -r  -p "  $label: " value
            fi
        fi
    fi
    if [[ -n "$value" ]]; then
        printf 'export %s=%q\n' "$var" "$value" >> "$SECRETS_FILE"
        echo "[install]   saved $var to .forge_env.secrets"
    else
        echo "[install]   $var left unset (skip)"
    fi
}

prompt_secret CAP_AGENT_NAME      "Agent display name shown in cap_ui (e.g. Mochi, 5090-bot)"   0
prompt_secret HF_TOKEN            "HuggingFace token (gated models; SAM3, etc.)"                 1
prompt_secret NVIDIA_API_KEY      "NVIDIA API key (default VLM backend; required for vision)"    1
prompt_secret GEMINI_API_KEY      "Gemini API key (optional VLM backend; blank to skip)"         1
prompt_secret ELEVENLABS_API_KEY  "ElevenLabs API key (voice output; blank to skip)"             1
prompt_secret ELEVENLABS_VOICE_ID "ElevenLabs Voice ID (required if ElevenLabs key set; blank to skip)" 0
# ─── Phase 11: generated runtime environment ──────────────────────────────
echo "[install] phase 11: writing .forge_env.runtime"
cat > "$ENV_SNIPPET" <<'EOF'
# Auto-generated by install/install_cap.sh. Do not edit or commit this file.
# Everything points at project-local third_party/ — no conda at runtime.
_ASPIRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANYGRASP_OPENSSL11_DIR="$_ASPIRE_ROOT/third_party/anygrasp_libs"
export OPENBLAS_HOME="$_ASPIRE_ROOT/third_party/anygrasp_libs"
# bundlesdf/__init__.py resolves runtime libs from this env var first; without
# it, it falls back to the LFS-tracked third_party/bundlesdf/libs/ (PCL 1.10),
# which mismatches the rebuilt 5090 closure (PCL 1.15).
export BUNDLESDF_RUNTIME_LIB_DIR="$_ASPIRE_ROOT/third_party/bundlesdf_5090"
export PATH="$_ASPIRE_ROOT/third_party/nodejs/bin:${PATH}"
# Put anygrasp_libs on LD_LIBRARY_PATH so Python can dlopen libportaudio.so.2,
# libssl.so.1.1, libopenblas.so.0 — and so the AnyGrasp launcher finds them.
export LD_LIBRARY_PATH="$_ASPIRE_ROOT/third_party/anygrasp_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# nvidia-curobo is an editable submodule with no git tags → setuptools-scm
# cannot detect a version. Pretend version so any `uv run` that re-resolves
# metadata (service panes etc.) doesn't trip LookupError mid-launch.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0
EOF

# ─── Final smoke test (fast) ───────────────────────────────────────────────
echo "[install] smoke-testing critical imports..."

# Idempotency check 1 — a second `uv sync` should be a no-op (<2s).
# Confirms phase 3 captured everything in pyproject.toml + uv.lock.
echo "[install]   re-running uv sync (expect no-op)..."
T0=$(date +%s%N)
CPATH="$AG_ENV/include${CPATH:+:$CPATH}" \
LIBRARY_PATH="$AG_ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0 \
    uv sync --extra cap --extra cap_tools --extra stt 2>&1 | tail -3
T1=$(date +%s%N); echo "[install]   uv sync re-run: $(( (T1 - T0) / 1000000 )) ms"

# Idempotency check 2 — cold python startup should be ~hundreds of ms.
T0=$(date +%s%N)
LD_LIBRARY_PATH="$ANYGRASP_LIBS_DIR" \
.venv/bin/python -c "print('python OK')" >/dev/null
T1=$(date +%s%N); echo "[install]   python startup: $(( (T1 - T0) / 1000000 )) ms"

T0=$(date +%s%N)
LD_LIBRARY_PATH="$ANYGRASP_LIBS_DIR" \
.venv/bin/python - <<'PY'
import importlib, logging, sys, warnings

# Silence harmless ROS-fallback noise from autolab_core (pulled in by graspnetAPI).
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# (module, expected version spec — string for display, None to skip check)
specs = [
    ("torch",          "==2.8.0"),
    ("transformers",   ">=5.1.0"),
    ("pyaudio",        None),
    ("faster_whisper", "==1.1.1"),
    ("graspnetAPI",    None),
    ("transforms3d",   ">=0.4"),
    ("curobo.geom",    None),
    ("pybind11",       None),
]

def pkg_version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", None) or "(no __version__)"
    except Exception:
        return None

print(f"  {'module':<18} {'installed':<18} {'expected':<12}")
errors = []
for mod, want in specs:
    v = pkg_version(mod)
    if v is None:
        errors.append(mod)
        print(f"  {mod:<18} {'FAILED':<18} {want or '':<12}")
    else:
        print(f"  {mod:<18} {v:<18} {want or 'any':<12}")
if errors:
    print(f"FAILED imports: {errors}")
    sys.exit(1)
print("imports OK")
PY
T1=$(date +%s%N); echo "[install]   import smoke-test: $(( (T1 - T0) / 1000000 )) ms"

cat <<EOF

============================================================================
  install_cap.sh complete.

  Runtime files (everything project-local):
    .venv/                          — Python deps via uv
    third_party/bundlesdf_5090/     — built BundleSDF + closure
    third_party/anygrasp_libs/      — OpenSSL 1.1, OpenBLAS, libportaudio + closure
    third_party/nodejs/             — Node.js prebuild
    .forge_env                      — operator configuration (never overwritten)
    .forge_env.runtime              — generated runtime paths
    .forge_env.secrets              — user secrets (preserved; gitignored)

  Conda envs (build-time only; deletable after this run):
    \$HOME/miniforge3/envs/bundlesdf-build       (BundleSDF C++/CUDA build deps)
    \$HOME/miniforge3/envs/aspire-anygrasp-libs   (openssl 1.1, openblas, portaudio)

  Launch the CAP service stack:
    bash tmux/launch_realworld_localserver_realsense.sh
============================================================================
EOF
