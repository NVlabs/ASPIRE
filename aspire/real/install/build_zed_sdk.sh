#!/usr/bin/env bash
# build_zed_sdk.sh — Install the Stereolabs ZED SDK so the project's locked
# pyzed wheel can load `libsl_zed.so` and talk to a connected ZED 2i.
#
# Run from repo root:  bash install/build_zed_sdk.sh
#
# Why this exists:
#   pyproject.toml pins pyzed to a specific Stereolabs URL (e.g. 5.2). That
#   wheel is just Cython bindings; it expects the matching native SDK at
#   /usr/local/zed/. install/install_cap.sh does NOT install the native SDK
#   because it requires sudo and writes outside the project tree. This script
#   is the one place that touches /usr/local/zed.
#
# What it does:
#   1. Preflight: confirms Ubuntu 22 + CUDA 12 host, reads the pyzed version
#      pinned in pyproject.toml so the SDK matches the wheel.
#   2. Downloads the matching ZED SDK .run from Stereolabs to /tmp.
#   3. Runs the installer as the regular user. The .run self-elevates with
#      sudo when needed; do NOT prefix this with sudo (it will refuse).
#   4. Re-runs `uv sync` so pyzed lands in .venv against the new SDK.
#   5. Smoke-tests: imports pyzed and enumerates connected cameras.
#
# Recommended answers when the installer prompts:
#   Install the CUDA toolkit?                   -> n   (already at /usr/local/cuda*)
#   Install the static libraries?               -> n
#   Install the Python API (pyzed)?             -> n   (uv sync handles it)
#   Install Object Detection / Body Tracking?   -> y   (small AI models)
#   Install AI samples / examples?              -> n
#   Install diagnostic tool?                    -> y   (ZED_Explorer for debugging)
#   Optimize NEURAL depth models?               -> y   (one-time, builds TRT engines)
#   Run ZED Diagnostic to optimize AI models?   -> n   (hours, JIT-built on first use)
#
# Does NOT touch:
#   - /usr/local/cuda*  (read-only)
#   - kernel modules, NVIDIA driver, nvidia-smi
#   - the project's .venv beyond `uv sync`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[36m[zed-sdk]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[zed-sdk ERROR]\033[0m %s\n' "$*" >&2; }

# Pre-cache sudo creds via a zenity popup so the .run installer's internal
# `sudo` calls don't block on the TTY (handy when running under tmux/pane
# splits, or via a launcher that doesn't expose the terminal). Falls back to
# the normal TTY prompt if no DISPLAY/zenity. tty_tickets means the cache
# applies to this shell's controlling TTY, which the .run inherits.
precache_sudo() {
    if sudo -n true 2>/dev/null; then
        log "  sudo creds already cached"
        return 0
    fi
    if [[ -n "${DISPLAY:-}" ]] && command -v zenity >/dev/null 2>&1; then
        log "  prompting for sudo password (GUI popup)"
        local askpass
        askpass="$(mktemp --suffix=.sh)"
        chmod 700 "$askpass"
        cat > "$askpass" <<'ASK'
#!/usr/bin/env bash
exec zenity --password --title="ZED SDK install — sudo password" 2>/dev/null
ASK
        SUDO_ASKPASS="$askpass" sudo -A -v
        local rc=$?
        rm -f "$askpass"
        return $rc
    fi
    log "  no DISPLAY/zenity; falling back to terminal sudo prompt"
    sudo -v
}

# ── Phase 0: preflight ────────────────────────────────────────────────────
log "phase 0: preflight"
[[ $EUID -ne 0 ]] || { err "do not run this script as root; the .run installer self-elevates with sudo"; exit 1; }
command -v wget >/dev/null  || { err "wget required";       exit 1; }
command -v sha256sum >/dev/null || { err "sha256sum required"; exit 1; }
[[ -x "$REPO_ROOT/.venv/bin/python" ]] || { err ".venv missing; run install/install_cap.sh first"; exit 1; }

UBUNTU_REL="$(lsb_release -rs 2>/dev/null || echo unknown)"
case "$UBUNTU_REL" in
    22.*) UBUNTU_TAG="ubuntu22" ;;
    24.*) UBUNTU_TAG="ubuntu24" ;;
    *) err "unsupported Ubuntu release: $UBUNTU_REL (this script targets 22.04 / 24.04)"; exit 1 ;;
esac
compgen -G "/usr/local/cuda-12*" >/dev/null || { err "CUDA 12.x not found at /usr/local/cuda-12*"; exit 1; }
log "  host: Ubuntu $UBUNTU_REL ($UBUNTU_TAG), CUDA 12.x present"

# Read the pyzed URL pinned in pyproject.toml and derive the SDK major/minor
# so the native SDK matches the wheel exactly (e.g. wheel 5.2 ↔ SDK 5.2.x).
PYZED_URL="$(grep -oE 'https://download.stereolabs.com/zedsdk/[^"[:space:]]+' "$REPO_ROOT/pyproject.toml" | head -1 || true)"
[[ -n "$PYZED_URL" ]] || { err "could not find pyzed URL in pyproject.toml"; exit 1; }
SDK_MAJMIN="$(printf '%s\n' "$PYZED_URL" | sed -nE 's|.*/zedsdk/([0-9]+\.[0-9]+)/.*|\1|p')"
[[ -n "$SDK_MAJMIN" ]] || { err "could not parse SDK version from $PYZED_URL"; exit 1; }
log "  project pin: pyzed $SDK_MAJMIN  ->  installing SDK $SDK_MAJMIN to match"

# ── Phase 1: short-circuit if already installed at the right version ──────
INSTALLED_VER=""
if [[ -x /usr/local/zed/tools/ZED_Diagnostic ]]; then
    INSTALLED_VER="$("$REPO_ROOT/.venv/bin/python" -c \
        "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())" 2>/dev/null || true)"
fi
if [[ -n "$INSTALLED_VER" && "$INSTALLED_VER" == ${SDK_MAJMIN}* ]]; then
    log "  SDK $INSTALLED_VER already installed and pyzed loads it; skipping install"
    SKIP_INSTALL=1
else
    SKIP_INSTALL=0
fi

# ── Phase 2: download + run installer ─────────────────────────────────────
RUN_FILE="/tmp/ZED_SDK_${UBUNTU_TAG}_cuda12.run"
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    [[ -n "${ZED_SDK_SHA256:-}" ]] || {
        err "ZED_SDK_SHA256 is required before downloading the vendor installer"
        err "obtain the checksum from an approved Stereolabs source"
        exit 1
    }
    log "phase 2: downloading SDK $SDK_MAJMIN .run -> $RUN_FILE"
    wget -O "$RUN_FILE" "https://download.stereolabs.com/zedsdk/${SDK_MAJMIN}/cu12/${UBUNTU_TAG}"
    printf '%s  %s\n' "$ZED_SDK_SHA256" "$RUN_FILE" | sha256sum -c -
    chmod +x "$RUN_FILE"
    log "  downloaded $(ls -lh "$RUN_FILE" | awk '{print $5}')"

    log "phase 3: running installer (interactive)"
    log "  answer the prompts as documented in the header of this script."
    log "  in particular: 'Install the Python API?' -> n  (uv sync handles it)"

    # Pop a GUI dialog for the sudo password up front so the .run's internal
    # `sudo` calls don't stall on the TTY mid-install.
    precache_sudo

    # Keep sudo cache warm during the long apt phase (the installer's many
    # internal sudo calls already refresh it, but this guarantees no stall).
    ( while sudo -n true 2>/dev/null; do sleep 60; done ) &
    SUDO_KEEPALIVE_PID=$!
    trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null || true' EXIT

    "$RUN_FILE"

    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    trap - EXIT

    [[ -e /usr/local/zed/lib/libsl_zed.so ]] || {
        err "installer finished but /usr/local/zed/lib/libsl_zed.so missing"
        exit 1
    }
fi

# ── Phase 4: install pyzed wheel only (--no-deps so nothing else moves) ───
# Critically: do NOT run `uv sync` here — that re-resolves the whole lockfile
# and on hosts with a stale uv.lock could shuffle unrelated packages. We just
# want pyzed bound to the new native SDK, so install only the wheel pinned in
# pyproject.toml without touching its declared deps (numpy/cython are already
# satisfied by `install_cap.sh` phase 3).
log "phase 4: uv pip install --no-deps pyzed $SDK_MAJMIN wheel"
if "$REPO_ROOT/.venv/bin/python" -c "import pyzed.sl as sl; \
    v=sl.Camera.get_sdk_version(); \
    import sys; sys.exit(0 if str(v).startswith('${SDK_MAJMIN}') else 1)" 2>/dev/null; then
    log "  pyzed already loads matching SDK; skipping wheel install"
else
    uv pip install --no-deps "$PYZED_URL" 2>&1 | tail -3
fi

# ── Phase 5: smoke test ───────────────────────────────────────────────────
log "phase 5: smoke test (load pyzed without printing device identifiers)"
"$REPO_ROOT/.venv/bin/python" - <<'PY'
import pyzed.sl as sl
print(f"  SDK version: {sl.Camera.get_sdk_version()}")
devs = list(sl.Camera.get_device_list())
print(f"  devices: {len(devs)}")
PY

# ── Phase 6: cleanup ──────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    rm -f "$RUN_FILE"
fi

cat <<EOF

============================================================================
  ZED SDK install complete ✓

  Native libs:  /usr/local/zed/
  Wheel:        pyzed $SDK_MAJMIN in $REPO_ROOT/.venv
  Diagnostic:   /usr/local/zed/tools/ZED_Explorer    (live preview)

  Keep any camera serial mapping in the local .forge_env.secrets file.

  Next: launch the stack with CAP_TOP_CAMERA_BACKEND=zed (already the default
  for 'top' in robot/camera_factory.py:48).
============================================================================
EOF
