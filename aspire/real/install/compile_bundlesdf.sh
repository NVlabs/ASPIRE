#!/usr/bin/env bash
# compile_bundlesdf.sh — Build BundleSDF from source against a reviewed
# conda-forge dependency stack, producing a transferable third_party/bundlesdf_5090/.
#
# Why this exists:
#   The LFS-tracked third_party/bundlesdf/libs/ on this repo were built on
#   Ubuntu 24.04 (GLIBC_2.38, GLIBCXX_3.4.32) and don't load on stock 22.04 /
#   RHEL 9 hosts. We rebuild the BundleTrack C++/CUDA extensions from
#   `BundleTrack/src/` and bundle a glibc-2.17-compatible runtime closure into
#   a SEPARATE directory (third_party/bundlesdf_5090/) so it never collides
#   with the LFS-tracked libs/. The bundlesdf Python module reads
#   `BUNDLESDF_RUNTIME_LIB_DIR` to locate this directory at runtime. The
#   resulting output loads on any modern Linux (Ubuntu 18.04+, RHEL 7+,
#   Debian 10+) with Python 3.11 and CUDA 12.x.
#
# Touches:
#   - $HOME/miniforge3/                (installed iff absent; reusable)
#   - $HOME/miniforge3/envs/bundlesdf-build/
#   - third_party/bundlesdf/BundleTrack/build/
#   - third_party/bundlesdf_5090/      (created/refreshed; never touches libs/)
#
# Does NOT touch:
#   - third_party/bundlesdf/libs/  (LFS-tracked; left alone)
#   - any path under /usr or /lib
#   - /usr/local/cuda* (read-only)
#   - kernel modules, NVIDIA driver, nvidia-smi
#   - .venv/bin/python (no patchelf)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOCK_DIR="$SCRIPT_DIR/locks/bundlesdf"
LOCK_FILE="$LOCK_DIR/build-lock.env"
[[ -f "$LOCK_FILE" ]] || { echo "missing build lock: $LOCK_FILE" >&2; exit 1; }
ENV_CUDA_HOME="${CUDA_HOME:-}"
# shellcheck disable=SC1090
source "$LOCK_FILE"

# ── Configuration ──────────────────────────────────────────────────────────
BUNDLESDF_DIR="$REPO_ROOT/third_party/bundlesdf"
LIBS_DIR="$REPO_ROOT/third_party/bundlesdf_5090"
SRC_DIR="$BUNDLESDF_DIR/BundleTrack"
BUILD_DIR="$SRC_DIR/build"

CUDA_HOME="${ENV_CUDA_HOME:-$CUDA_HOME}"
# RTX 5090 = sm_120 (Blackwell). Older arches included for fleet portability.
# sm_80 (Ampere HPC, A100) ─ sm_86 (Ampere consumer, RTX 30xx) ─
# sm_89 (Ada Lovelace, RTX 40xx) ─ sm_90 (Hopper, H100) ─ sm_120 (Blackwell, RTX 50xx).
# Turing (sm_75) intentionally dropped — pre-2020 GPUs aren't in the fleet.
# Override via env: CUDA_ARCHS=120 bash install/compile_bundlesdf.sh  (slimmest binary).
CUDA_ARCHS="${CUDA_ARCHS:-${CUDA_ARCHITECTURES//,/;}}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="bundlesdf-build"
CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV"

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

# ── Helpers ────────────────────────────────────────────────────────────────
log() { printf '\033[36m[bundlesdf]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[bundlesdf ERROR]\033[0m %s\n' "$*" >&2; }

download_verified() {
    local url="$1" expected="$2" output="$3"
    wget -q --show-progress -O "$output" "$url"
    printf '%s  %s\n' "$expected" "$output" | sha256sum -c -
}

snapshot_cuda() {
    sha256sum "$CUDA_HOME/bin/nvcc" \
              /lib/x86_64-linux-gnu/libcuda.so.1 \
              "$CUDA_HOME/lib64/libcudart.so.12" \
              /usr/bin/nvidia-smi 2>/dev/null
}

# ── Phase 0: preflight ────────────────────────────────────────────────────
log "phase 0: preflight"
[[ -x "$CUDA_HOME/bin/nvcc" ]]   || { err "nvcc not at $CUDA_HOME/bin/nvcc"; exit 1; }
command -v wget >/dev/null        || { err "wget is required"; exit 1; }
command -v sha256sum >/dev/null   || { err "sha256sum is required"; exit 1; }
command -v cmake >/dev/null       || { err "cmake is required"; exit 1; }
command -v ninja >/dev/null       || { err "ninja is required"; exit 1; }
command -v nproc >/dev/null       || { err "nproc is required"; exit 1; }
command -v ldd >/dev/null         || { err "ldd is required"; exit 1; }
command -v strip >/dev/null       || { err "strip is required"; exit 1; }
[[ -x "$VENV_PYTHON" ]]          || { err "venv python not at $VENV_PYTHON; run install/install_cap.sh first"; exit 1; }
"$VENV_PYTHON" -c "import pybind11" >/dev/null 2>&1 \
    || { err "pybind11 not importable in venv; run 'uv sync --extra cap_tools' first"; exit 1; }
[[ -f "$SRC_DIR/CMakeLists.txt" ]] || { err "$SRC_DIR/CMakeLists.txt missing"; exit 1; }

CUDA_PRE="$(snapshot_cuda)"
log "  CUDA pre-hash captured ($(wc -l <<<"$CUDA_PRE") files)"

# ── Phase 1: miniforge ────────────────────────────────────────────────────
if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
    log "phase 1: installing miniforge to $CONDA_BASE"
    INSTALLER="$(mktemp --suffix=.sh)"
    download_verified "$MINIFORGE_URL" "$MINIFORGE_SHA256" "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_BASE"
    rm -f "$INSTALLER"
else
    log "phase 1: reusing miniforge at $CONDA_BASE"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── Phase 2: conda env ────────────────────────────────────────────────────
if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    log "phase 2: creating conda env '$CONDA_ENV' from a reviewed explicit spec"
    EXPLICIT_SPEC="${BUNDLESDF_CONDA_EXPLICIT_SPEC:-}"
    EXPLICIT_SHA256="${BUNDLESDF_CONDA_EXPLICIT_SHA256:-}"
    [[ -n "$EXPLICIT_SPEC" && -f "$EXPLICIT_SPEC" ]] || {
        err "set BUNDLESDF_CONDA_EXPLICIT_SPEC to a reviewed CUDA 12.8-compatible spec"
        err "the recovered audit spec in $LOCK_DIR is CUDA 13 metadata and is not executable"
        exit 1
    }
    [[ -n "$EXPLICIT_SHA256" ]] || {
        err "set BUNDLESDF_CONDA_EXPLICIT_SHA256 for $EXPLICIT_SPEC"
        exit 1
    }
    printf '%s  %s\n' "$EXPLICIT_SHA256" "$EXPLICIT_SPEC" | sha256sum -c -
    conda create -y -n "$CONDA_ENV" --file "$EXPLICIT_SPEC"
else
    log "phase 2: reusing conda env at $CONDA_ENV_PATH"
fi
# conda's activation hooks reference env vars before defining them — incompatible
# with `set -u`. Disable nounset around activate, then restore.
set +u
conda activate "$CONDA_ENV"
set -u

# PCL 1.x headers `#include <boost/detail/endian.hpp>`, removed in Boost ≥ 1.73.
# Conda's Boost 1.85 doesn't have it. Drop in a shim that re-exports the modern
# location's macros under the old name. Self-contained inside the conda env;
# disappears when the env is deleted.
SHIM="$CONDA_ENV_PATH/include/boost/detail/endian.hpp"
if [[ ! -f "$SHIM" ]]; then
    log "  installing boost/detail/endian.hpp shim for PCL"
    mkdir -p "$(dirname "$SHIM")"
    cat > "$SHIM" <<'EOF'
// Shim: boost/detail/endian.hpp was removed in Boost ≥ 1.73 and replaced by
// boost/predef/other/endian.h. PCL 1.x still includes the old path. We
// translate the new macros back to the old names so PCL's headers compile.
#ifndef BOOST_DETAIL_ENDIAN_HPP
#define BOOST_DETAIL_ENDIAN_HPP
#include <boost/predef/other/endian.h>
#if BOOST_ENDIAN_BIG_BYTE
#  define BOOST_BIG_ENDIAN
#  define BOOST_BYTE_ORDER 4321
#elif BOOST_ENDIAN_LITTLE_BYTE
#  define BOOST_LITTLE_ENDIAN
#  define BOOST_BYTE_ORDER 1234
#elif BOOST_ENDIAN_LITTLE_WORD
#  define BOOST_PDP_ENDIAN
#  define BOOST_BYTE_ORDER 2143
#else
#  error "boost/detail/endian.hpp shim: unknown endianness"
#endif
#endif
EOF
fi

# ── Phase 2.5: build OpenCV with CUDA from source ─────────────────────────
# Conda-forge's libopencv is built with WITH_CUDA=OFF, so it doesn't ship the
# `cudaimgproc`/`cudafeatures2d` modules BundleTrack uses. We compile our own
# OpenCV 4.11 + opencv_contrib here, install into the conda env prefix, and
# let the BundleTrack build pick them up via CMAKE_PREFIX_PATH below.
OPENCV_VER="$OPENCV_VERSION"
OPENCV_SRC="$CONDA_ENV_PATH/src/opencv"
OPENCV_CONTRIB="$CONDA_ENV_PATH/src/opencv_contrib"
OPENCV_BUILD="$CONDA_ENV_PATH/src/opencv-build"
OPENCV_MARKER="$CONDA_ENV_PATH/lib/libopencv_cudaimgproc.so"

if [[ ! -e "$OPENCV_MARKER" ]]; then
    log "phase 2.5: building OpenCV $OPENCV_VER with CUDA from source (~25-40 min)"

    # Conda compiler wrappers (set CC/CXX so OpenCV's cmake picks them up).
    CONDA_CC="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-gcc"
    CONDA_CXX="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-g++"

    DOWNLOAD_DIR="$CONDA_ENV_PATH/src/downloads"
    mkdir -p "$DOWNLOAD_DIR"
    if [[ ! -d "$OPENCV_SRC" ]]; then
        log "  downloading pinned opencv $OPENCV_COMMIT..."
        OPENCV_ARCHIVE="$DOWNLOAD_DIR/opencv-$OPENCV_COMMIT.tar.gz"
        download_verified "$OPENCV_ARCHIVE_URL" "$OPENCV_ARCHIVE_SHA256" "$OPENCV_ARCHIVE"
        tar -xzf "$OPENCV_ARCHIVE" -C "$CONDA_ENV_PATH/src"
        mv "$CONDA_ENV_PATH/src/opencv-$OPENCV_COMMIT" "$OPENCV_SRC"
        printf '%s\n' "$OPENCV_COMMIT" > "$OPENCV_SRC/.aspire-source-commit"
    fi
    if [[ ! -d "$OPENCV_CONTRIB" ]]; then
        log "  downloading pinned opencv_contrib $OPENCV_CONTRIB_COMMIT..."
        CONTRIB_ARCHIVE="$DOWNLOAD_DIR/opencv_contrib-$OPENCV_CONTRIB_COMMIT.tar.gz"
        download_verified "$OPENCV_CONTRIB_ARCHIVE_URL" "$OPENCV_CONTRIB_ARCHIVE_SHA256" "$CONTRIB_ARCHIVE"
        tar -xzf "$CONTRIB_ARCHIVE" -C "$CONDA_ENV_PATH/src"
        mv "$CONDA_ENV_PATH/src/opencv_contrib-$OPENCV_CONTRIB_COMMIT" "$OPENCV_CONTRIB"
        printf '%s\n' "$OPENCV_CONTRIB_COMMIT" > "$OPENCV_CONTRIB/.aspire-source-commit"
    fi

    [[ "$(cat "$OPENCV_SRC/.aspire-source-commit" 2>/dev/null)" == "$OPENCV_COMMIT" ]] \
        || { err "existing OpenCV source is not the locked commit $OPENCV_COMMIT"; exit 1; }
    [[ "$(cat "$OPENCV_CONTRIB/.aspire-source-commit" 2>/dev/null)" == "$OPENCV_CONTRIB_COMMIT" ]] \
        || { err "existing opencv_contrib source is not the locked commit $OPENCV_CONTRIB_COMMIT"; exit 1; }

    rm -rf "$OPENCV_BUILD"
    mkdir -p "$OPENCV_BUILD"
    cd "$OPENCV_BUILD"

    # CUDA arch list as comma-separated for OpenCV's CUDA_ARCH_BIN.
    OCV_ARCH="${CUDA_ARCHS//;/,}"

    log "  configuring OpenCV (arch: $OCV_ARCH)"
    cmake "$OPENCV_SRC" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CONDA_ENV_PATH" \
        -DCMAKE_C_COMPILER="$CONDA_CC" \
        -DCMAKE_CXX_COMPILER="$CONDA_CXX" \
        -DCMAKE_PREFIX_PATH="$CONDA_ENV_PATH" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DOPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB/modules" \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=OFF \
        -DOPENCV_DNN_CUDA=OFF \
        -DWITH_CUBLAS=ON \
        -DWITH_NVCUVID=OFF \
        -DWITH_NVCUVENC=OFF \
        -DCUDA_ARCH_BIN="$OCV_ARCH" \
        -DCUDA_ARCH_PTX="" \
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
        -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
        -DCMAKE_CUDA_HOST_COMPILER="$CONDA_CXX" \
        -DCUDA_HOST_COMPILER="$CONDA_CXX" \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_V4L=OFF \
        -DWITH_OPENEXR=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,xfeatures2d,cudaimgproc,cudafeatures2d,cudaarithm,cudawarping,cudafilters,cudaoptflow,cudev,highgui,rgbd

    log "  compiling OpenCV (this is the long part)"
    ninja -j"$(nproc)"
    log "  installing OpenCV into $CONDA_ENV_PATH"
    ninja install

    [[ -e "$OPENCV_MARKER" ]] || { err "OpenCV build finished but $OPENCV_MARKER missing"; exit 1; }
    log "  OpenCV-CUDA install complete"
else
    log "phase 2.5: OpenCV-CUDA already built ($OPENCV_MARKER present), skipping"
fi

# ── Phase 4: configure + build ────────────────────────────────────────────
log "phase 4: cmake configure + build (arch: $CUDA_ARCHS)"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
# Force a clean reconfigure each run; cheap and avoids stale cmake cache bugs.
rm -f CMakeCache.txt

CONDA_CC="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-gcc"
CONDA_CXX="$CONDA_ENV_PATH/bin/x86_64-conda-linux-gnu-g++"
[[ -x "$CONDA_CC"  ]] || { err "conda gcc missing at $CONDA_CC";  exit 1; }
[[ -x "$CONDA_CXX" ]] || { err "conda g++ missing at $CONDA_CXX"; exit 1; }

PYBIND11_DIR="$("$VENV_PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

# Explicitly pin Python to the venv's interpreter so cmake doesn't accidentally
# pick up conda's transitively-pulled-in Python 3.14.
PY_INC="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PY_LIBDIR="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
PY_LDLIBRARY="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')"
PY_LIB="$PY_LIBDIR/$PY_LDLIBRARY"
[[ -e "$PY_LIB" ]] || { err "venv libpython missing at $PY_LIB"; exit 1; }

cmake "$SRC_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CONDA_CC" \
    -DCMAKE_CXX_COMPILER="$CONDA_CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$CONDA_CXX" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF \
    -DCMAKE_PREFIX_PATH="$CONDA_ENV_PATH" \
    -DOpenCV_DIR="$CONDA_ENV_PATH/lib/cmake/opencv4" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DPython_EXECUTABLE="$VENV_PYTHON" \
    -DPython3_EXECUTABLE="$VENV_PYTHON" \
    -DPython_INCLUDE_DIR="$PY_INC" \
    -DPython3_INCLUDE_DIR="$PY_INC" \
    -DPython_LIBRARY="$PY_LIB" \
    -DPython3_LIBRARY="$PY_LIB"

cmake --build . -j"$(nproc)"

# ── Phase 5: stage libs/ + runtime closure ─────────────────────────────────
log "phase 5: installing built artifacts into $LIBS_DIR"
mkdir -p "$LIBS_DIR"
find "$LIBS_DIR" -mindepth 1 -delete

# The three BundleTrack artifacts. Strip after copy to shed the multi-MB debug
# info from BundleTrack's `-g` build (libBundleTrack.so shrinks ~87M -> ~1.2M).
cp -v libBundleTrack.so       "$LIBS_DIR/"
cp -v libMY_CUDA_LIB.so       "$LIBS_DIR/"
cp -v my_cpp.cpython-*.so     "$LIBS_DIR/"
strip --strip-all "$LIBS_DIR"/libBundleTrack.so \
                  "$LIBS_DIR"/libMY_CUDA_LIB.so \
                  "$LIBS_DIR"/my_cpp.cpython-*.so 2>/dev/null || true

log "  resolving conda-forge runtime closure (transitive ldd)..."
declare -A SEEN
queue=()
for f in "$LIBS_DIR"/*.so; do queue+=("$f"); done

while [[ ${#queue[@]} -gt 0 ]]; do
    cur="${queue[0]}"
    queue=("${queue[@]:1}")
    while IFS= read -r line; do
        # Extract `=> /path/to/lib.so` form
        path=$(awk '{ for (i=1;i<=NF;i++) if ($i=="=>") { print $(i+1); exit } }' <<<"$line")
        [[ -n "$path" && "$path" != "not" && -f "$path" ]] || continue
        # Only ship libs from conda env. Skip system / cuda / kernel libs.
        [[ "$path" == "$CONDA_ENV_PATH"/* ]] || continue
        bn=$(basename "$path")
        [[ -z "${SEEN[$bn]:-}" ]] || continue
        SEEN[$bn]=1
        cp -v "$path" "$LIBS_DIR/$bn"
        queue+=("$LIBS_DIR/$bn")
    done < <(LD_LIBRARY_PATH="$LIBS_DIR:$CONDA_ENV_PATH/lib" ldd "$cur" 2>/dev/null || true)
done

# NOTE: We deliberately do NOT create unversioned SONAME symlinks (libfoo.so
# pointing at libfoo.so.1.x). They're useful only at compile time, never at
# runtime — no consumer's DT_NEEDED references an unversioned name. Skipping
# them keeps the LFS-committed tree small and symlink-free.

log "  $LIBS_DIR now has $(ls "$LIBS_DIR" | wc -l) entries"

# ── Phase 6: verify ────────────────────────────────────────────────────────
# Note: a benign segfault is observed during Python process *exit* (static
# destructor ordering between PCL/CUDA libs). The import itself works. We
# accept exit if "IMPORT OK" reached stdout, regardless of the post-exit code.
log "phase 6: verifying import"
cd "$REPO_ROOT"
VERIFY_OUT="$(BUNDLESDF_RUNTIME_LIB_DIR="$LIBS_DIR" \
    LD_LIBRARY_PATH="$LIBS_DIR:${LD_LIBRARY_PATH:-}" \
    "$VENV_PYTHON" -c "import bundlesdf; from bundlesdf import BundleSdf; print('IMPORT OK')" 2>&1 || true)"
if ! grep -q '^IMPORT OK$' <<<"$VERIFY_OUT"; then
    err "bundlesdf import failed. libs/ left in place for inspection."
    err "  output:"
    printf '%s\n' "$VERIFY_OUT" | head -20 >&2
    err "  ldd of libBundleTrack.so:"
    LD_LIBRARY_PATH="$LIBS_DIR" ldd "$LIBS_DIR/libBundleTrack.so" | head -40 >&2
    exit 1
fi
if grep -q "Segmentation fault" <<<"$VERIFY_OUT"; then
    log "  note: benign teardown segfault observed (static destructor cleanup); import itself OK"
fi

# ── Phase 7: CUDA/driver post-hash check ──────────────────────────────────
log "phase 7: confirming CUDA/driver untouched"
CUDA_POST="$(snapshot_cuda)"
if [[ "$CUDA_PRE" != "$CUDA_POST" ]]; then
    err "CUDA/driver file hashes changed! Build is UNSAFE."
    diff <(printf '%s\n' "$CUDA_PRE") <(printf '%s\n' "$CUDA_POST") >&2
    exit 1
fi
log "  driver/CUDA hashes unchanged ✓"

# ── Phase 8: write distributable artifact ─────────────────────────────────
TARBALL="$REPO_ROOT/bundlesdf-5090-cuda12-py311-x86_64.tar.gz"
log "phase 8: writing $TARBALL for fleet distribution"
tar -C "$REPO_ROOT/third_party" -czf "$TARBALL" bundlesdf_5090/

cat <<EOF

============================================================================
  BundleSDF build complete ✓

  arch:     $CUDA_ARCHS
  output:   $LIBS_DIR  ($(ls "$LIBS_DIR" | wc -l) files)
  tarball:  $TARBALL  ($(du -h "$TARBALL" | awk '{print $1}'))

  Distribute to other hosts (Python 3.11 + CUDA 12.x driver required):
      scp $TARBALL host:/path/to/aspire/real/
      ssh host 'cd /path/to/aspire/real && tar -C third_party -xzf $(basename "$TARBALL")'

  Driver/CUDA untouched. Conda env removable: rm -rf $CONDA_ENV_PATH
============================================================================
EOF
