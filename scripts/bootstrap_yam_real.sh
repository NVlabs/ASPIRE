#!/usr/bin/env bash
# Self-contained Real YAM station runtime. Builds <repo>/.venv-yam-real with the
# adapter, station clients, and the artifact recorder's image codec only (the
# yam-real extra = portal + imageio). It deliberately does NOT install the
# model-serving providers (SAM3/GraspGen/PyRoki/cuRobo) -- those run from their
# own per-provider venvs and are reached over HTTP/Portal. Host prerequisites are
# NOT installed here -- NVIDIA drivers, SocketCAN, USB and the arm udev rules.
# They are written down in docs/host-setup.md; this comment used to call them
# "documented" when nothing in the tree described them.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/venv_backing.sh"
source "$REPO_ROOT/scripts/lib/fingerprint.sh"
PYTHON_BIN="${CAP_HARNESS_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for _cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$_cand" >/dev/null 2>&1; then PYTHON_BIN="$_cand"; break; fi
  done
fi
STATION="${CAP_HARNESS_YAM_STATION:-yam-example}"

command -v uv >/dev/null || { printf 'error: uv is required\n' >&2; exit 1; }
[[ -f "$REPO_ROOT/uv.lock" ]] || { printf 'error: uv.lock is missing\n' >&2; exit 1; }

VENV_ROOT="$(resolve_canonical_venv "$REPO_ROOT" ".venv-yam-real")"
UV_CACHE_ROOT="${CAP_HARNESS_YAM_CACHE:-$REPO_ROOT/.uv-cache}"

# No-op re-bootstrap when the resolved deps are unchanged.
FINGERPRINT="$(compute_venv_fingerprint "$REPO_ROOT" "${CAP_HARNESS_PROFILE:-rtx5090}" "runtime:yam-real:yam-real")"
if venv_fingerprint_current "$VENV_ROOT" "$FINGERPRINT"; then
  printf 'yam-real runtime is up to date (fingerprint %s); skipping rebuild\n' "${FINGERPRINT:0:12}"
  exit 0
fi

mkdir -p "$UV_CACHE_ROOT"
UV_CACHE_DIR="$UV_CACHE_ROOT" uv venv --allow-existing --python "$PYTHON_BIN" "$VENV_ROOT"
# Client-only station runtime: yam-real (portal + imageio) + dev; no providers.
VIRTUAL_ENV="$VENV_ROOT" UV_PROJECT_ENVIRONMENT="$VENV_ROOT" UV_CACHE_DIR="$UV_CACHE_ROOT" \
  uv sync --frozen --active --project "$REPO_ROOT" \
    --extra yam-real --extra dev

# Verify the runtime is self-contained: cap-harness + portal import, the station
# profile + immutable calibration load, and a fake-station passive doctor passes
# with zero follower command RPCs -- all without touching hardware or services.
"$VENV_ROOT/bin/python" - "$STATION" <<'PY'
import sys

import numpy as np
import portal  # noqa: F401 - the yam-real transport dependency must be importable

from cap_harness.yam_real.adapter import YamRealAdapter
from cap_harness.yam_real.cameras import SyntheticCamera
from cap_harness.yam_real.config import load_yam_station_config
from cap_harness.yam_real.env import RealYamEnv

station = sys.argv[1]
cfg = load_yam_station_config(station)


class _FakeArm:
    """Answers observations; records any command so the count can be asserted."""

    commands = 0

    def get_observations(self):
        return {
            "joint_pos": np.zeros(6),
            "joint_vel": np.zeros(7),
            "gripper_pos": np.zeros(1),
        }

    def command_joint_state(self, state):
        type(self).commands += 1

    def close(self):
        pass


# A camera is required, not optional: reset() returns an Observation and the
# shared contract needs at least one camera in it. Without this the check died
# on "the station camera produced no frame" before testing anything.
env = RealYamEnv(cfg, {"left": _FakeArm(), "right": _FakeArm()}, camera=SyntheticCamera())
adapter = YamRealAdapter(env)  # unauthorized: the default

state = adapter.get_robot_state()
assert set(state.joint_positions) == {"left", "right"}, state.joint_positions
assert state.joint_positions["left"].shape == (6,), state.joint_positions["left"].shape

# Reset moves the robot, so an unauthorized one must observe without commanding.
# It returns an Observation, not a result -- the refusal is visible in the arms
# never being touched, which is the property worth asserting anyway. The old
# check read `result.ok` on an Observation and could never have passed.
observation = adapter.reset(seed=0)
assert observation.cameras, "reset must return an observation carrying a camera"
assert _FakeArm.commands == 0, f"unauthorized reset issued {_FakeArm.commands} commands"
assert adapter.command_rpc_count == 0, adapter.command_rpc_count

# And the interlock itself, on a path that does return a typed result.
refused = adapter.go_home()
assert not refused.ok, "unauthorized go_home must be refused"
assert refused.error.code.value == "safety_interlock", refused.error.code
assert _FakeArm.commands == 0, f"refused go_home issued {_FakeArm.commands} commands"

print(
    f"Verified yam-real runtime: station={station} "
    f"calibration={cfg.calibration.bundle_id} "
    f"passive reads=ok interlock=refused command_rpc_count=0"
)
PY

write_venv_fingerprint "$VENV_ROOT" "$FINGERPRINT" "runtime:yam-real:yam-real" "${CAP_HARNESS_PROFILE:-rtx5090}"
cat <<EOF
yam-real bootstrap complete.
  environment: $VENV_ROOT
  station:     $STATION
  extras:      yam-real (portal + imageio) + dev  (no SAM3/GraspGen/PyRoki/cuRobo)

This is the client runtime: it talks to arm servers over portal but carries no
CAN stack. To drive motors from this machine, add the server extra. Both env
vars are required: without them uv builds a fresh ./.venv, exits 0, and leaves
this runtime untouched -- the missing damiao_motor then surfaces at bus.connect(),
i.e. the moment you energize the motors.
  VIRTUAL_ENV="$VENV_ROOT" UV_PROJECT_ENVIRONMENT="$VENV_ROOT" \\
    uv sync --frozen --active --project "$REPO_ROOT" \\
    --extra yam-real --extra yam-real-server --extra dev
  "$VENV_ROOT/bin/yam-servers" left  --station $STATION
  "$VENV_ROOT/bin/yam-servers" right --station $STATION
EOF
