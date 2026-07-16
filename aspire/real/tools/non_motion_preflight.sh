#!/usr/bin/env bash
# Read-only YAM workstation inventory. This script does not start services,
# connect to arm RPC endpoints, bring CAN links up/down, or send CAN frames.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

section() {
  printf '\n[%s]\n' "$1"
}

section safety
echo "physical_motion=disabled"
echo "arm_rpc_queries=disabled"
echo "can_transmit=disabled"

section host
sed -nE 's/^(PRETTY_NAME|VERSION_ID)=/\1=/p' /etc/os-release 2>/dev/null
uname -srmo 2>/dev/null || true
"$PYTHON" --version 2>&1 || true
uv --version 2>/dev/null || true

section gpu_cuda
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader 2>&1 || true
else
  echo "nvidia_smi=not_found"
fi
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  "$CUDA_HOME/bin/nvcc" --version 2>/dev/null | tail -1
else
  echo "nvcc=not_found"
fi

section realsense
"$PYTHON" - <<'PY' 2>&1 || true
try:
    import pyrealsense2 as rs
except Exception as exc:
    print(f"pyrealsense2=unavailable ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

try:
    from importlib.metadata import version
    print(f"pyrealsense2={version('pyrealsense2')}")
except Exception:
    print("pyrealsense2=installed_version_unknown")

try:
    devices = list(rs.context().query_devices())
except Exception as exc:
    print(f"enumeration=failed ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

print(f"device_count={len(devices)}")
for index, device in enumerate(devices):
    def info(field):
        try:
            return device.get_info(field)
        except Exception:
            return "unknown"

    # Deliberately omit rs.camera_info.serial_number and physical_port.
    print(
        f"device[{index}].name={info(rs.camera_info.name)} "
        f"firmware={info(rs.camera_info.firmware_version)} "
        f"recommended_firmware={info(rs.camera_info.recommended_firmware_version)}"
    )
PY

section zed
"$PYTHON" - <<'PY' 2>&1 || true
try:
    from importlib.metadata import version
    print(f"pyzed={version('pyzed')}")
except Exception as exc:
    print(f"pyzed=unavailable ({type(exc).__name__}: {exc})")

try:
    import pyzed.sl as sl
    sdk_version = str(sl.Camera.get_sdk_version()).strip()
    print(f"native_sdk={sdk_version if sdk_version else 'not_detected'}")
except Exception as exc:
    print(f"native_sdk=unavailable ({type(exc).__name__}: {exc})")
PY

section arm_software
"$PYTHON" - <<'PY' 2>&1 || true
from importlib.metadata import PackageNotFoundError, version

for package in ("damiao-motor", "i2rt", "python-can"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=not_installed")

print("motor_firmware=not_queried_non_motion_policy")
PY

section can_mapping
for interface in can_follow_l can_follow_r can_leader_l can_leader_r; do
  if [[ -e "/sys/class/net/$interface" ]]; then
    state="$(cat "/sys/class/net/$interface/operstate" 2>/dev/null || echo unknown)"
    echo "$interface=$state"
  else
    echo "$interface=absent"
  fi
done
echo "expected_bitrate=1000000"

section local_rules
for rule in /etc/udev/rules.d/99-can.rules /etc/udev/rules.d/99-realsense.rules; do
  if [[ -f "$rule" ]]; then
    echo "$(basename "$rule")=present"
  else
    echo "$(basename "$rule")=absent"
  fi
done

section result
echo "preflight_complete=true"
